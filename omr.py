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

# Ajusta esto a tu PDF (zona donde están las burbujas)
# IMPORTANTE: mejor que sobre y luego dividir por filas reales
OMR_REGION = {"y0": 520, "y1": 3000, "x0": 520, "x1": 1600}

# Umbrales (robustos para escáner + móvil)
UMBRAL_VACIO = 0.055
UMBRAL_DOBLE_RATIO = 0.78
UMBRAL_DOBLE_ABS = 0.040

# ROI interior para no contar el borde impreso
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
# NORMALIZAR A4 (marcas negras en esquinas)
# ============================================================
def normalizar_a4_con_marcas(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # negro -> blanco
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
        if 0.65 < aspect < 1.35:  # casi cuadrado
            cx = x + bw / 2
            cy = y + bh / 2
            candidates.append((cx, cy))

    # fallback: al menos reescalado
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
# QR ROBUSTO (multi intentos + ROI + rotaciones + preprocess)
# ============================================================
def _try_decode_qr(detector, img_bgr):
    # detectAndDecodeMulti si existe
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
    variants.append(img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE (contraste)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    variants.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # Sharpen
    kernel_sh = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sh = cv2.filter2D(gray, -1, kernel_sh)
    variants.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    # Otsu + invertida
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # Adaptive + invertida
    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    variants.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return variants


def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # 1) Imagen completa
    data = _try_decode_qr(det, img_bgr)
    if data:
        return data, None

    # 2) ROI sup-izq (tu QR está ahí)
    qr_roi = img_bgr[0:1500, 0:1500].copy()

    for scale in [1.0, 1.6, 2.2, 3.0, 3.8]:
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

    # 3) zona superior más grande
    top = img_bgr[0:1700, 0:1700].copy()
    for v in _preprocess_variants(top):
        data = _try_decode_qr(det, v)
        if data:
            return data, b64jpg(qr_roi)

    return None, b64jpg(qr_roi)


def parsear_codigo_qr(codigo):
    """
    ✅ Soporta estos formatos:

    A) id_examen|id_alumno|num_preguntas|pagina?
    B) id_examen|id_alumno|fecha|num_preguntas|pagina?

    En tu caso real suele ser B) (con fecha en la 3ª parte).
    """
    if not codigo:
        return None

    partes = [p.strip() for p in codigo.split("|")]
    if len(partes) < 3:
        return None

    # helper
    def _is_int(s):
        try:
            int(s)
            return True
        except:
            return False

    try:
        id_examen = int(partes[0])
        id_alumno = int(partes[1])
    except:
        return None

    fecha = None
    num_preguntas = None
    pagina = 1

    # Caso A: tercero es num_preguntas
    if len(partes) >= 3 and _is_int(partes[2]):
        num_preguntas = int(partes[2])
        if len(partes) >= 4 and _is_int(partes[3]):
            pagina = int(partes[3])

    # Caso B: tercero es fecha y cuarto num_preguntas
    elif len(partes) >= 4 and _is_int(partes[3]):
        fecha = partes[2]
        num_preguntas = int(partes[3])
        if len(partes) >= 5 and _is_int(partes[4]):
            pagina = int(partes[4])

    if not num_preguntas or num_preguntas <= 0:
        return None

    if pagina not in (1, 2):
        pagina = 1

    return id_examen, id_alumno, fecha, num_preguntas, pagina


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
def filas_a_leer(num_preguntas, pagina, max_por_hoja=30):
    if pagina == 1:
        return min(num_preguntas, max_por_hoja), 0
    if num_preguntas <= max_por_hoja:
        return 0, max_por_hoja
    return min(num_preguntas - max_por_hoja, max_por_hoja), max_por_hoja


# ============================================================
# DETECTOR OMR (divide por filas reales)
# ============================================================
def detectar_respuestas(zona_bin, filas, debug_a4=None):
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape

    # margen lateral pequeño
    margen = int(w * 0.05)
    zona = zona_bin[:, margen:w - margen]
    h2, w2 = zona.shape

    # ✅ clave: dividir por filas reales, no por 30 fijo
    alto_fila = max(1, int(h2 / float(filas)))
    ancho_op = max(1, int(w2 / 4.0))

    respuestas = []

    for i in range(filas):
        y0 = i * alto_fila
        y1 = min(h2, (i + 1) * alto_fila)
        fila = zona[y0:y1, :]

        dens = []
        rois = []

        for j in range(4):
            x0 = j * ancho_op
            x1 = min(w2, (j + 1) * ancho_op)
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
        second = sorted_d[1] if len(sorted_d) > 1 else 0.0

        if max_d < UMBRAL_VACIO:
            resp = ""
        elif (second > UMBRAL_DOBLE_ABS) and (second > max_d * UMBRAL_DOBLE_RATIO):
            resp = "X"   # doble marca
        else:
            resp = OPCIONES[idx]

        respuestas.append(resp)

        # debug
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

    return respuestas, debug_a4


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) QR ANTES DE NORMALIZAR (a veces mejor)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) normalizar A4
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR DESPUÉS si el anterior falló
    codigo, debug_qr = (codigo0, debug_qr0)
    parsed = parsed0

    if not parsed:
        codigo, debug_qr = leer_qr_robusto(img_a4)
        parsed = parsear_codigo_qr(codigo) if codigo else None

    if not parsed:
        return {
            "ok": False,
            "error": "QR no detectado o formato QR inválido",
            "codigo": codigo or "",
            "debug_qr": debug_qr
        }

    id_examen, id_alumno, fecha, num_preguntas, pagina = parsed

    # 3) binarizar tinta
    th = binarizar_tinta_pro(img_a4)

    # 4) recorte OMR
    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]

    # debug overlay sobre A4
    debug_a4 = img_a4.copy()

    # 5) filas
    filas, offset = filas_a_leer(num_preguntas, pagina, max_por_hoja=30)
    if filas <= 0:
        return {
            "ok": False,
            "error": "Esta página no tiene preguntas según el QR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha": fecha,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr
        }

    # 6) detectar respuestas
    respuestas_lista, debug_a4 = detectar_respuestas(zona_bin, filas, debug_a4)

    # salida dict por número global
    respuestas = {}
    for i, r in enumerate(respuestas_lista, start=1):
        respuestas[str(offset + i)] = r

    return {
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha": fecha,
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
