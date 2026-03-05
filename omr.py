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

# ✅ Mantengo TU OMR_REGION (la que te estaba acercando)
OMR_REGION = {
    "y0": 520,
    "y1": 3150,
    "x0": 620,
    "x1": 1680
}

MAX_FILAS_POR_HOJA = 30

# Umbrales (robustos para móvil + escáner)
UMBRAL_VACIO = 0.050
UMBRAL_DOBLE_RATIO = 0.80
UMBRAL_DOBLE_ABS = 0.035

# ROI interior para ignorar borde impreso
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
# 1) NORMALIZAR A4 con marcas negras (TU VERSION ROBUSTA)
# ============================================================
def normalizar_a4_con_marcas(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)

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
        if 0.70 < aspect < 1.30:
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return cv2.resize(img_bgr, (A4_W, A4_H))

    candidates.sort(key=lambda t: t[2], reverse=True)
    pts = np.array([[c[0], c[1]] for c in candidates], dtype=np.float32)

    targets = {
        "tl": np.array([0.0, 0.0], dtype=np.float32),
        "tr": np.array([float(w), 0.0], dtype=np.float32),
        "br": np.array([float(w), float(h)], dtype=np.float32),
        "bl": np.array([0.0, float(h)], dtype=np.float32),
    }

    chosen = {}
    used = set()

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
# 2) QR ROBUSTO (TU VERSION QUE SÍ LEÍA)
# ============================================================
def _try_decode(det, img_bgr):
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

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    out.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    k = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]], dtype=np.float32)
    sh = cv2.filter2D(g1, -1, k)
    out.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    _, otsu = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    ad = cv2.adaptiveThreshold(g1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    out.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return out


def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # intento imagen completa
    for v in _variants(img_bgr):
        s = _try_decode(det, v)
        if s:
            return s, None

    # ROI fijo arriba izquierda
    qr_roi = img_bgr[0:1200, 0:1200].copy()
    debug_qr = b64jpg(qr_roi, 90)

    scales = [1.0, 1.8, 2.6, 3.4, 4.2]
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

    big = img_bgr[0:1600, 0:1600].copy()
    for v in _variants(big):
        s = _try_decode(det, v)
        if s:
            return s, debug_qr

    return None, debug_qr


def parsear_codigo_qr(codigo):
    if not codigo:
        return None

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

    # Fallback: extraer enteros aunque el QR tenga nombre/fecha
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
# 3) BINARIZACIÓN PRO
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
# 4) FILAS SEGÚN QR + PÁGINA
# ============================================================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# ============================================================
# 5) NUEVO: detectar centros de filas (ADAPTATIVO)
# ============================================================
def detectar_centros_filas(zona_bin, filas_esperadas):
    """
    Busca picos horizontales (tinta de los círculos) para centrar cada fila.
    Si falla, cae a rejilla fija.
    """
    h, w = zona_bin.shape

    prof = np.sum(zona_bin > 0, axis=1).astype(np.float32)

    k = 31
    kernel = np.ones(k, dtype=np.float32) / k
    prof_s = np.convolve(prof, kernel, mode="same")

    peaks = []
    prof_work = prof_s.copy()

    # distancia mínima entre filas
    min_dist = max(22, int(h / max(1, MAX_FILAS_POR_HOJA) * 0.65))

    for _ in range(filas_esperadas):
        idx = int(np.argmax(prof_work))
        if prof_work[idx] <= 0:
            break
        peaks.append(idx)
        y0 = max(0, idx - min_dist)
        y1 = min(h, idx + min_dist)
        prof_work[y0:y1] = 0

    peaks = sorted(peaks)

    # Si detecta muy pocos, fallback a rejilla fija
    if len(peaks) < max(3, int(filas_esperadas * 0.6)):
        step = h / float(MAX_FILAS_POR_HOJA)
        peaks = [int((i + 0.5) * step) for i in range(filas_esperadas)]

    return peaks


# ============================================================
# 6) DETECTOR OMR (MEJORADO: filas adaptativas)
# ============================================================
def detectar_respuestas(zona_bin, filas, debug_a4=None):
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape

    margen = int(w * 0.05)
    zona = zona_bin[:, margen:w - margen]
    h2, w2 = zona.shape

    ancho_op = int(w2 / 4.0)

    # ✅ centros de filas reales
    centers = detectar_centros_filas(zona, filas)

    # ventana alrededor del centro
    win = max(18, int(h2 / MAX_FILAS_POR_HOJA * 0.55))

    respuestas = []

    for i in range(filas):
        cy = centers[i]
        y0 = max(0, cy - win)
        y1 = min(h2, cy + win)
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

            cv2.putText(
                debug_a4,
                f"{i+1}:{resp or '-'}",
                (OMR_REGION["x0"] - 180, OMR_REGION["y0"] + (i + 1) * int(h2 / MAX_FILAS_POR_HOJA) - 10),
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

    # 0) QR antes de normalizar
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) Normalizar A4
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR después si falló
    codigo, debug_qr = codigo0, debug_qr0
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

    # 3) binarización
    th = binarizar_tinta_pro(img_a4)

    # 4) recorte OMR
    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]

    debug_a4 = img_a4.copy()

    # dibuja el rectángulo OMR_REGION para depurar
    cv2.rectangle(
        debug_a4,
        (OMR_REGION["x0"], OMR_REGION["y0"]),
        (OMR_REGION["x1"], OMR_REGION["y1"]),
        (255, 0, 0),
        3
    )

    # 5) filas según QR
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

    # 6) detectar respuestas (mejorado)
    respuestas_lista, debug_a4 = detectar_respuestas(zona_bin, filas, debug_a4)

    # 7) dict global
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
