from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# =========================
# CONFIG PLANTILLA
# =========================
A4_W, A4_H = 2480, 3508           # A4 normalizado ~300dpi
OPCIONES = ["A", "B", "C", "D"]

# Zona aproximada donde está el bloque de burbujas (A4 normalizado)
# (No hace falta que sea milimétrica: ahora recortamos verticalmente automático)
OMR_REGION = {"y0": 520, "y1": 2750, "x0": 620, "x1": 1350}

# Máximo filas por hoja (tu regla: hasta 30 por hoja; 31-60 en hoja 2)
MAX_FILAS_POR_HOJA = 30

# Umbrales OMR (ajustables)
UMBRAL_VACIO = 0.055
UMBRAL_DOBLE_RATIO = 0.78
UMBRAL_DOBLE_ABS = 0.040

# ROI interior dentro de cada casilla para ignorar borde impreso
INNER_PAD_X = 0.22
INNER_PAD_Y = 0.22


# ============================================================
# UTIL: encode image -> base64 jpg
# ============================================================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


# ============================================================
# NORMALIZAR A4 (marcas negras)
# ============================================================
def normalizar_a4_con_marcas(img_bgr):
    """
    Detecta 4 marcas negras de esquina y corrige perspectiva a A4.
    Fallback: resize a A4 si no encuentra 4.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # binaria invertida (negro -> blanco)
    _, th = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh else 0
        # marcas casi cuadradas
        if 0.65 < aspect < 1.35:
            cx = x + bw / 2
            cy = y + bh / 2
            candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return cv2.resize(img_bgr, (A4_W, A4_H))

    pts = np.array([[c[0], c[1]] for c in candidates], dtype=np.float32)

    # ordenar por suma/diff (clásico)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL

    dst = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))
    return warped


# ============================================================
# QR ROBUSTO (muchos intentos + rotaciones + preprocesado)
# ============================================================
def _try_decode_qr(detector, img_bgr):
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img_bgr)
        if ok and decoded_info:
            for s in decoded_info:
                s = (s or "").strip()
                if s:
                    return s
    except Exception:
        pass

    data, _, _ = detector.detectAndDecode(img_bgr)
    data = (data or "").strip()
    return data if data else None


def _preprocess_variants(img_bgr):
    variants = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(img_bgr)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    variants.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # Sharpen
    kernel_sh = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sh = cv2.filter2D(gray, -1, kernel_sh)
    variants.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    # Otsu + invert
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # Adaptive + invert
    ad = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )
    variants.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return variants


def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # 1) Full
    data = _try_decode_qr(det, img_bgr)
    if data:
        return data, None

    # 2) ROI sup-izq (tu QR)
    qr_roi = img_bgr[0:1500, 0:1500].copy()

    for scale in [1.0, 1.7, 2.3, 3.0, 3.6]:
        roi = qr_roi
        if scale != 1.0:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        rots = [
            roi,
            cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(roi, cv2.ROTATE_180),
            cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        for r in rots:
            for v in _preprocess_variants(r):
                data = _try_decode_qr(det, v)
                if data:
                    return data, b64jpg(qr_roi)

    # 3) Zona top grande (por si el ROI quedó corto)
    top = img_bgr[0:1900, 0:1900].copy()
    for v in _preprocess_variants(top):
        data = _try_decode_qr(det, v)
        if data:
            return data, b64jpg(qr_roi)

    return None, b64jpg(qr_roi)


def parsear_codigo_qr(codigo):
    """
    Formato esperado:
      id_examen|id_alumno|num_preguntas|pagina
    pagina opcional (1 o 2)
    """
    if not codigo:
        return None

    partes = codigo.split("|")
    if len(partes) < 3:
        return None

    try:
        id_examen = int(partes[0])
        id_alumno = int(partes[1])
        num_preguntas = int(partes[2])

        pagina = 1
        if len(partes) >= 4 and str(partes[3]).strip().isdigit():
            pagina = int(partes[3])
        pagina = 1 if pagina not in (1, 2) else pagina

        return id_examen, id_alumno, num_preguntas, pagina
    except Exception:
        return None


# ============================================================
# BINARIZACIÓN PRO (móvil + escáner)
# ============================================================
def binarizar_tinta_pro(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # contraste
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
# NUEVO ✅ Recorte vertical automático (quita espacio vacío)
# ============================================================
def ajustar_zona_vertical(zona_bin):
    """
    Detecta automáticamente donde empiezan/terminan las burbujas
    usando densidad de tinta por fila.
    """
    h, w = zona_bin.shape
    dens = np.sum(zona_bin > 0, axis=1)

    if dens.size == 0:
        return zona_bin

    mx = float(np.max(dens))
    if mx <= 1:
        return zona_bin

    umbral = mx * 0.12  # sensible
    filas_validas = np.where(dens > umbral)[0]

    if len(filas_validas) < 10:
        return zona_bin

    y0 = max(0, int(filas_validas[0]) - 25)
    y1 = min(h, int(filas_validas[-1]) + 25)

    if y1 - y0 < int(h * 0.25):
        return zona_bin

    return zona_bin[y0:y1, :]


# ============================================================
# FILAS SEGÚN QR + PÁGINA
# ============================================================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    # página 2
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# ============================================================
# DETECTOR OMR ✅ (altura por "filas reales", no por 30 fijas)
# ============================================================
def detectar_respuestas(zona_bin, filas, debug_a4=None, region_rect=None):
    """
    zona_bin: binaria recortada SOLO al bloque de burbujas (vertical ya ajustado)
    filas: número de preguntas reales en esta hoja
    region_rect: (x0,y0,x1,y1) en coords A4 para pintar rectángulos (opcional)
    """
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape

    # recorte lateral para evitar bordes
    margen = int(w * 0.06)
    if w - 2 * margen > 50:
        zona = zona_bin[:, margen:w - margen]
    else:
        zona = zona_bin

    h2, w2 = zona.shape
    alto_fila = int(h2 / float(filas))         # ✅ CLAVE: filas reales
    ancho_op = int(w2 / 4.0)

    respuestas = []

    for i in range(filas):
        y0 = i * alto_fila
        y1 = (i + 1) * alto_fila
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

        max_d = max(dens)
        idx = int(np.argmax(dens))
        sorted_d = sorted(dens, reverse=True)
        second = sorted_d[1]

        if max_d < UMBRAL_VACIO:
            resp = ""
        elif (second > UMBRAL_DOBLE_ABS) and (second > max_d * UMBRAL_DOBLE_RATIO):
            resp = "X"
        else:
            resp = OPCIONES[idx]

        respuestas.append(resp)

        # Debug overlay
        if debug_a4 is not None and region_rect is not None:
            rx0, ry0, rx1, ry1 = region_rect
            # convertir coords zona->A4
            for j in range(4):
                cx0, cy0, cx1, cy1 = rois[j]

                # volver a sumar margen lateral si se aplicó
                if w - 2 * margen > 50:
                    cx0 += margen
                    cx1 += margen

                X0 = rx0 + cx0
                X1 = rx0 + cx1
                Y0 = ry0 + cy0
                Y1 = ry0 + cy1

                cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 255, 0), 2)

                if resp and resp != "X" and OPCIONES[j] == resp:
                    cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 0, 255), 3)

            # texto de la pregunta
            cv2.putText(
                debug_a4,
                f"{i+1}:{resp or '-'}",
                (rx0 - 140, ry0 + (i + 1) * alto_fila - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if resp == "X" else (0, 0, 0),
                2
            )

    return respuestas, debug_a4


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) QR antes de normalizar (a veces se lee mejor)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) Normalizar A4
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR después de normalizar (si falló antes)
    codigo, debug_qr = (codigo0, debug_qr0)
    parsed = parsed0

    if not parsed:
        codigo, debug_qr = leer_qr_robusto(img_a4)
        parsed = parsear_codigo_qr(codigo) if codigo else None

    if not parsed:
        return {
            "ok": False,
            "error": "QR no detectado",
            "debug_qr": debug_qr
        }

    id_examen, id_alumno, num_preguntas, pagina = parsed

    # 3) Binarizar tinta
    th = binarizar_tinta_pro(img_a4)

    # 4) Recorte OMR aproximado
    x0, x1 = OMR_REGION["x0"], OMR_REGION["x1"]
    y0, y1 = OMR_REGION["y0"], OMR_REGION["y1"]

    zona_bin_raw = th[y0:y1, x0:x1].copy()
    if zona_bin_raw.size == 0:
        return {"ok": False, "error": "OMR_REGION fuera de imagen A4"}

    # ✅ Recorte vertical automático (la gran mejora)
    zona_bin = ajustar_zona_vertical(zona_bin_raw)

    # Para debug binario en jpg
    zona_bin_vis = (zona_bin.copy())
    if zona_bin_vis.ndim == 2:
        zona_bin_vis = cv2.cvtColor(zona_bin_vis, cv2.COLOR_GRAY2BGR)

    # Para que el overlay cuadre, necesitamos saber cuánto recortamos arriba
    # Calculamos el offset Y dentro de zona_bin_raw
    # (lo recomputamos igual que en ajustar_zona_vertical)
    dens = np.sum(zona_bin_raw > 0, axis=1)
    mx = float(np.max(dens)) if dens.size else 0
    y_crop = 0
    if mx > 1:
        umbral = mx * 0.12
        filas_validas = np.where(dens > umbral)[0]
        if len(filas_validas) >= 10:
            y0c = max(0, int(filas_validas[0]) - 25)
            y1c = min(zona_bin_raw.shape[0], int(filas_validas[-1]) + 25)
            if y1c - y0c >= int(zona_bin_raw.shape[0] * 0.25):
                y_crop = y0c

    # debug overlay A4
    debug_a4 = img_a4.copy()

    # 5) Filas según QR/página
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

    # region_rect en coords A4 para pintar:
    region_rect = (x0, y0 + y_crop, x1, y1)

    # 6) Detectar respuestas
    respuestas_lista, debug_a4 = detectar_respuestas(
        zona_bin,
        filas,
        debug_a4=debug_a4,
        region_rect=region_rect
    )

    # 7) Respuestas como dict global (1..60)
    respuestas = {}
    for i, r in enumerate(respuestas_lista, start=1):
        respuestas[str(offset + i)] = r

    # 8) Debug images base64
    return {
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "num_preguntas": num_preguntas,
        "pagina": pagina,
        "respuestas": respuestas,
        "debug_image": b64jpg(debug_a4, 85),
        "debug_qr": debug_qr,
        "debug_bin": b64jpg(zona_bin_vis, 85)
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
