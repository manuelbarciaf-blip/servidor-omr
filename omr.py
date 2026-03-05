from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import re

app = Flask(__name__)

# ============================================================
# CONFIG ✅ (AJUSTADA A TU HOJA REAL)
# ============================================================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

# 🔧 OMR_REGION ampliada (tu grid se veía "apretado" / recorte corto)
# Ajusta solo si cambias tu plantilla PDF.
OMR_REGION = {
    "y0": 520,     # antes 650
    "y1": 3150,    # antes 3000
    "x0": 620,     # antes 780
    "x1": 1680     # antes 1450
}

# La plantilla tiene slots base de 30 por hoja
MAX_FILAS_POR_HOJA = 30

# Umbrales (robustos para móvil + escáner)
UMBRAL_VACIO = 0.050          # baja un poco para bolígrafo azul
UMBRAL_DOBLE_RATIO = 0.80
UMBRAL_DOBLE_ABS = 0.035

# ROI interior dentro de cada celda para ignorar el borde impreso
INNER_PAD_X = 0.23
INNER_PAD_Y = 0.23


# ============================================================
# UTIL: imagen -> base64 jpg
# ============================================================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


# ============================================================
# 1) NORMALIZAR A4 con marcas negras (más robusto)
# ============================================================
def normalizar_a4_con_marcas(img_bgr):
    """
    Detecta 4 marcas negras en esquinas y aplica perspectiva a A4.
    - Filtra por área + forma cuadrada.
    - Evita seleccionar la misma marca para dos esquinas.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binaria invertida: negro => blanco
    _, th = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)

    # Limpieza
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2200:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh <= 0 or bw <= 0:
            continue
        aspect = bw / float(bh)
        # cuadrado aproximado
        if 0.70 < aspect < 1.30:
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            candidates.append((cx, cy, area))

    # Fallback: solo escala a A4
    if len(candidates) < 4:
        return cv2.resize(img_bgr, (A4_W, A4_H))

    # Ordena por área DESC (prioriza marcas más grandes)
    candidates.sort(key=lambda t: t[2], reverse=True)
    pts = np.array([[c[0], c[1]] for c in candidates], dtype=np.float32)

    # Targets de esquinas
    targets = {
        "tl": np.array([0.0, 0.0], dtype=np.float32),
        "tr": np.array([float(w), 0.0], dtype=np.float32),
        "br": np.array([float(w), float(h)], dtype=np.float32),
        "bl": np.array([0.0, float(h)], dtype=np.float32),
    }

    chosen = {}
    used = set()

    # Selecciona el mejor para cada esquina, sin repetir índice
    for name, t in targets.items():
        best_i = None
        best_d = None
        for i in range(len(pts)):
            if i in used:
                continue
            d = float(np.sum((pts[i] - t) ** 2))
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        if best_i is not None:
            chosen[name] = pts[best_i]
            used.add(best_i)

    if len(chosen) < 4:
        return cv2.resize(img_bgr, (A4_W, A4_H))

    src = np.array([chosen["tl"], chosen["tr"], chosen["br"], chosen["bl"]], dtype=np.float32)
    dst = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))
    return warped


# ============================================================
# 2) QR ROBUSTO (multi intentos + ROI fijo + rotaciones)
# ============================================================
def _try_decode(det, img_bgr):
    # Multi (si existe)
    try:
        ok, decoded_info, points, _ = det.detectAndDecodeMulti(img_bgr)
        if ok and decoded_info:
            for s in decoded_info:
                s = (s or "").strip()
                if s:
                    return s
    except Exception:
        pass

    s, _, _ = det.detectAndDecode(img_bgr)
    s = (s or "").strip()
    return s if s else None


def _variants(img_bgr):
    out = []
    out.append(img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    out.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # Sharpen
    k = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]], dtype=np.float32)
    sh = cv2.filter2D(g1, -1, k)
    out.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    # Otsu bin
    _, otsu = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # Adaptive bin
    ad = cv2.adaptiveThreshold(g1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    out.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return out


def leer_qr_robusto(img_bgr):
    """
    Devuelve (texto_qr or None, debug_qr_base64)
    """
    det = cv2.QRCodeDetector()

    # Intento 0: imagen completa
    for v in _variants(img_bgr):
        s = _try_decode(det, v)
        if s:
            return s, None

    # ROI fijo arriba izquierda (tu QR está SIEMPRE ahí)
    # Nota: en A4 normalizado el QR está en el margen sup izq dentro del marco negro.
    qr_roi = img_bgr[0:1200, 0:1200].copy()
    debug_qr = b64jpg(qr_roi, 90)

    scales = [1.0, 1.8, 2.6, 3.4]
    rots = [
        lambda im: im,
        lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
        lambda im: cv2.rotate(im, cv2.ROTATE_180),
        lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    for sc in scales:
        roi = qr_roi
        if sc != 1.0:
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        for rot in rots:
            rr = rot(roi)
            for v in _variants(rr):
                s = _try_decode(det, v)
                if s:
                    return s, debug_qr

    # Último intento: ROI más grande (más contexto del marco)
    big = img_bgr[0:1600, 0:1600].copy()
    for v in _variants(big):
        s = _try_decode(det, v)
        if s:
            return s, debug_qr

    return None, debug_qr


def parsear_codigo_qr(codigo):
    """
    ✅ QR imprescindible. Permitimos:
      - "id_examen|id_alumno|num_preguntas|pagina"
      - o texto con números mezclados (sacamos los 3-4 primeros enteros)
    """
    if not codigo:
        return None

    # Preferencia: separador |
    if "|" in codigo:
        partes = [p.strip() for p in codigo.split("|") if p.strip()]
        if len(partes) >= 3:
            try:
                id_examen = int(partes[0])
                id_alumno = int(partes[1])
                num_preg = int(partes[2])
                pagina = int(partes[3]) if len(partes) >= 4 and partes[3].isdigit() else 1
                pagina = 1 if pagina not in (1, 2) else pagina
                return id_examen, id_alumno, num_preg, pagina
            except:
                pass

    # Fallback: extraer enteros por regex (muy útil si el QR trae nombre/fecha)
    nums = re.findall(r"\d+", codigo)
    if len(nums) < 3:
        return None

    try:
        id_examen = int(nums[0])
        id_alumno = int(nums[1])
        num_preg = int(nums[2])
        pagina = int(nums[3]) if len(nums) >= 4 else 1
        pagina = 1 if pagina not in (1, 2) else pagina
        return id_examen, id_alumno, num_preg, pagina
    except:
        return None


# ============================================================
# 3) BINARIZACIÓN PRO (móvil/escáner)
# ============================================================
def binarizar_tinta_pro(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return th


# ============================================================
# 4) FILAS SEGÚN QR + PÁGINA (hasta 60 => 2 hojas)
# ============================================================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# ============================================================
# 5) DETECTOR OMR (rejilla estable 1 columna A-D)
# ============================================================
def detectar_respuestas(zona_bin, filas, debug_a4=None):
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape

    # margen lateral pequeño (evita contornos del marco)
    margen = int(w * 0.05)
    zona = zona_bin[:, margen:w - margen]
    h2, w2 = zona.shape

    # Slot base (30 filas)
    alto_slot = int(h2 / float(MAX_FILAS_POR_HOJA))
    ancho_op = int(w2 / 4.0)

    respuestas = []

    for i in range(filas):
        y0 = i * alto_slot
        y1 = (i + 1) * alto_slot
        fila = zona[y0:y1, :]

        dens = []
        rois = []

        for j in range(4):
            x0 = j * ancho_op
            x1 = (j + 1) * ancho_op
            celda = fila[:, x0:x1]

            if celda.size == 0:
                dens.append(0.0)
                rois.append((x0, y0, x1, y1))
                continue

            ih, iw = celda.shape
            pad_x = int(iw * INNER_PAD_X)
            pad_y = int(ih * INNER_PAD_Y)

            inner = celda[pad_y:ih - pad_y, pad_x:iw - pad_x]
            if inner.size == 0:
                inner = celda

            score = cv2.countNonZero(inner) / float(inner.size)
            dens.append(score)
            rois.append((x0, y0, x1, y1))

        max_d = float(max(dens))
        idx = int(np.argmax(dens))
        second = float(sorted(dens, reverse=True)[1])

        if max_d < UMBRAL_VACIO:
            resp = ""
        elif (second > UMBRAL_DOBLE_ABS) and (second > max_d * UMBRAL_DOBLE_RATIO):
            resp = "X"
        else:
            resp = OPCIONES[idx]

        respuestas.append(resp)

        # Debug overlay
        if debug_a4 is not None:
            for j in range(4):
                x0, yy0, x1, yy1 = rois[j]
                X0 = OMR_REGION["x0"] + margen + x0
                X1 = OMR_REGION["x0"] + margen + x1
                Y0 = OMR_REGION["y0"] + yy0
                Y1 = OMR_REGION["y0"] + yy1

                cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 255, 0), 2)
                if resp and resp != "X" and OPCIONES[j] == resp:
                    cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 0, 255), 3)

            # etiqueta
            cv2.putText(
                debug_a4,
                f"{i+1}:{resp or '-'}",
                (OMR_REGION["x0"] - 180, OMR_REGION["y0"] + (i + 1) * alto_slot - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if resp == "X" else (0, 0, 0),
                2
            )

    return respuestas, debug_a4


# ============================================================
# PIPELINE PRINCIPAL ✅ (QR imprescindible)
# ============================================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) Intento QR ANTES de normalizar (a veces se lee mejor así)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) Normalizar por marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) Intento QR DESPUÉS de normalizar (si el anterior falló)
    codigo, debug_qr = codigo0, debug_qr0
    parsed = parsed0
    if not parsed:
        codigo, debug_qr = leer_qr_robusto(img_a4)
        parsed = parsear_codigo_qr(codigo) if codigo else None

    if not parsed:
        # QR imprescindible
        return {
            "ok": False,
            "error": "QR no detectado",
            "debug_qr": debug_qr  # para verlo en PHP
        }

    id_examen, id_alumno, num_preguntas, pagina = parsed

    # 3) Binarización tinta
    th = binarizar_tinta_pro(img_a4)

    # 4) Recorte zona OMR
    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]

    # Debug overlay A4
    debug_a4 = img_a4.copy()
    # dibuja el rectángulo OMR REGION para ver si está bien
    cv2.rectangle(
        debug_a4,
        (OMR_REGION["x0"], OMR_REGION["y0"]),
        (OMR_REGION["x1"], OMR_REGION["y1"]),
        (255, 0, 0),
        3
    )

    # 5) Filas de esta hoja
    filas, offset = filas_a_leer(num_preguntas, pagina)
    if filas <= 0:
        return {
            "ok": False,
            "error": "Esta página no tiene preguntas según el QR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr
        }

    # 6) Detectar respuestas
    respuestas_lista, debug_a4 = detectar_respuestas(zona_bin, filas, debug_a4)

    # 7) dict global (1..60)
    respuestas = {}
    for i, r in enumerate(respuestas_lista, start=1):
        respuestas[str(offset + i)] = r

    return {
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "num_preguntas": num_preguntas,
        "pagina": pagina,
        "respuestas": respuestas,
        "debug_image": b64jpg(debug_a4, 85),
        "debug_qr": debug_qr
    }


# ============================================================
# ENDPOINTS
# ============================================================
@app.route("/corregir_omr", methods=["POST"])
def corregir_omr():
    if "imagen" not in request.files:
        return jsonify({"ok": False, "error": "Falta imagen"}), 400

    binario = request.files["imagen"].read()
    res = procesar_omr(binario)
    return jsonify(res)


@app.route("/")
def home():
    return "Servidor OMR ✅ (/corregir_omr)"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
