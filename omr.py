from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# =========================
# CONFIG PLANTILLA
# =========================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

OMR_REGION = {"y0": 650, "y1": 3000, "x0": 780, "x1": 1450}
MAX_FILAS_POR_HOJA = 30

UMBRAL_VACIO = 0.055
UMBRAL_DOBLE_RATIO = 0.78
UMBRAL_DOBLE_ABS = 0.040

# ROI interior dentro de cada casilla para no contar el borde impreso
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

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
        if 0.65 < aspect < 1.35:
            cx = x + bw / 2
            cy = y + bh / 2
            candidates.append((cx, cy))

    if len(candidates) < 4:
        return cv2.resize(img_bgr, (A4_W, A4_H))

    pts = np.array(candidates, dtype=np.float32)

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
    # intenta multi si existe
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
    """Genera variantes para que el QRDetector tenga más opciones."""
    variants = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(img_bgr)

    # CLAHE (mejor contraste)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    variants.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # Sharpen
    kernel_sh = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sh = cv2.filter2D(gray, -1, kernel_sh)
    variants.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # Adaptive
    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    variants.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return variants


def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # 1) Intento directo (imagen completa)
    data = _try_decode_qr(det, img_bgr)
    if data:
        return data, None

    # 2) ROI superior izquierda (tu QR está ahí)
    qr_roi = img_bgr[0:1300, 0:1300].copy()

    # Probar escalas
    for scale in [1.0, 1.8, 2.5, 3.2]:
        roi = qr_roi
        if scale != 1.0:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Probar rotaciones (móvil)
        rots = [
            roi,
            cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(roi, cv2.ROTATE_180),
            cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        for r in rots:
            # Variantes preprocess
            for v in _preprocess_variants(r):
                data = _try_decode_qr(det, v)
                if data:
                    return data, b64jpg(qr_roi)

    # 3) Último intento: buscar QR en zona grande superior (más contexto)
    top = img_bgr[0:1600, 0:1600].copy()
    for v in _preprocess_variants(top):
        data = _try_decode_qr(det, v)
        if data:
            return data, b64jpg(qr_roi)

    return None, b64jpg(qr_roi)


def parsear_codigo_qr(codigo):
    """
    Formato esperado:
      id_examen|id_alumno|num_preguntas|pagina
    pagina opcional (1 o 2).
    """
    partes = codigo.split("|")
    if len(partes) < 3:
        return None
    try:
        id_examen = int(partes[0])
        id_alumno = int(partes[1])
        num_preguntas = int(partes[2])
        pagina = int(partes[3]) if len(partes) >= 4 and str(partes[3]).strip().isdigit() else 1
        pagina = 1 if pagina not in (1, 2) else pagina
        return id_examen, id_alumno, num_preguntas, pagina
    except:
        return None


# ============================================================
# BINARIZACIÓN PRO
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
# FILAS SEGÚN QR + PÁGINA
# ============================================================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# ============================================================
# DETECTOR OMR (rejilla estable, 1 columna A-D)
# ============================================================
def detectar_respuestas(zona_bin, filas, debug_a4=None):
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape

    margen = int(w * 0.06)
    zona = zona_bin[:, margen:w - margen]
    h2, w2 = zona.shape

    alto_fila = int(h2 / float(MAX_FILAS_POR_HOJA))
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

        if debug_a4 is not None:
            # pintar rectángulos por opción
            for j in range(4):
                x0, yy0, x1, yy1 = rois[j]

                X0 = OMR_REGION["x0"] + margen + x0
                X1 = OMR_REGION["x0"] + margen + x1
                Y0 = OMR_REGION["y0"] + yy0
                Y1 = OMR_REGION["y0"] + yy1

                cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 255, 0), 2)

                if resp and resp != "X" and OPCIONES[j] == resp:
                    cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 0, 255), 3)

    return respuestas, debug_a4


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) QR ANTES DE NORMALIZAR (a veces se lee mejor así)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) normalizar A4
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR DESPUÉS DE NORMALIZAR (si el anterior falló)
    codigo, debug_qr = (codigo0, debug_qr0)
    parsed = parsed0

    if not parsed:
        codigo, debug_qr = leer_qr_robusto(img_a4)
        parsed = parsear_codigo_qr(codigo) if codigo else None

    if not parsed:
        return {
            "ok": False,
            "error": "QR no detectado",
            "debug_qr": debug_qr,  # para que lo veas en PHP
        }

    id_examen, id_alumno, num_preguntas, pagina = parsed

    # 3) binarizar
    th = binarizar_tinta_pro(img_a4)

    # 4) recorte OMR
    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]

    # debug overlay sobre A4
    debug_a4 = img_a4.copy()

    # 5) filas
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

    # 6) detectar respuestas
    respuestas_lista, debug_a4 = detectar_respuestas(zona_bin, filas, debug_a4)

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
