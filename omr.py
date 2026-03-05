from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# =========================
# CONFIG
# =========================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

MAX_FILAS_POR_HOJA = 30  # en tu diseño: máximo 30 por página (1..30) y (31..60)

# Umbrales (ajustables)
UMBRAL_VACIO = 0.060
UMBRAL_DOBLE_RATIO = 0.80
UMBRAL_DOBLE_ABS = 0.045

# Para medir relleno en círculos (porcentaje del radio para contar tinta)
RADIO_INTERIOR_RATIO = 0.55

# ROI QR (siempre arriba izquierda)
QR_ROI = (0, 0, 1350, 1350)  # (y0, x0, y1, x1)

# ROI aproximado donde están las burbujas (A4 normalizado)
# (solo para limitar la búsqueda de círculos; la rejilla final se calcula con círculos)
BUBBLES_SEARCH_ROI = (250, 520, 2700, 1500)  # y0, x0, y1, x1


# =========================
# Helpers
# =========================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


def rotate_variants(img_bgr):
    return [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


# =========================
# Perspectiva A4 por marcas negras (mejorado por cuadrantes)
# =========================
def normalizar_a4_con_marcas(img_bgr):
    """
    Detecta 4 marcas negras (esquinas) y corrige perspectiva a A4.
    Selección robusta: busca candidatos cuadrados y elige 1 por cuadrante (TL/TR/BL/BR).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Negro -> blanco
    _, th = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0:
            continue
        aspect = bw / float(bh)
        # cuadrados aproximados
        if 0.70 < aspect < 1.30:
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            cands.append((cx, cy, area))

    if len(cands) < 4:
        return cv2.resize(img_bgr, (A4_W, A4_H))

    # Elegir mejor candidato por cuadrante: máximo area y más cercano a esquina
    def pick(quadrant):
        if quadrant == "tl":
            tx, ty = 0, 0
            filt = [(cx, cy, a) for (cx, cy, a) in cands if cx < w * 0.5 and cy < h * 0.5]
        elif quadrant == "tr":
            tx, ty = w, 0
            filt = [(cx, cy, a) for (cx, cy, a) in cands if cx >= w * 0.5 and cy < h * 0.5]
        elif quadrant == "bl":
            tx, ty = 0, h
            filt = [(cx, cy, a) for (cx, cy, a) in cands if cx < w * 0.5 and cy >= h * 0.5]
        else:
            tx, ty = w, h
            filt = [(cx, cy, a) for (cx, cy, a) in cands if cx >= w * 0.5 and cy >= h * 0.5]

        if not filt:
            return None

        best = None
        best_score = None
        for (cx, cy, a) in filt:
            d2 = (cx - tx) ** 2 + (cy - ty) ** 2
            # score: prioriza area pero también cercanía
            score = d2 / max(a, 1.0)
            if best_score is None or score < best_score:
                best_score = score
                best = (cx, cy)
        return best

    tl = pick("tl")
    tr = pick("tr")
    bl = pick("bl")
    br = pick("br")

    if not all([tl, tr, bl, br]):
        return cv2.resize(img_bgr, (A4_W, A4_H))

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))
    return warped


# =========================
# QR súper robusto (OpenCV QRCodeDetector)
# =========================
def _try_decode(det, img_bgr):
    try:
        ok, decoded, _, _ = det.detectAndDecodeMulti(img_bgr)
        if ok and decoded:
            for s in decoded:
                s = (s or "").strip()
                if s:
                    return s
    except Exception:
        pass

    s, _, _ = det.detectAndDecode(img_bgr)
    s = (s or "").strip()
    return s if s else None


def _variants_for_qr(img_bgr):
    out = []
    out.append(img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    out.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # Sharpen
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sh = cv2.filter2D(gray, -1, k)
    out.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    # Otsu (y su invertida)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # Adaptive (y su invertida)
    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    out.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return out


def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # 1) Imagen completa con rotaciones
    for im in rotate_variants(img_bgr):
        s = _try_decode(det, im)
        if s:
            return s, None

    # 2) ROI arriba izquierda (tu QR está ahí)
    y0, x0, y1, x1 = QR_ROI
    roi0 = img_bgr[y0:y1, x0:x1].copy()
    debug_qr = b64jpg(roi0)

    scales = [1.0, 1.6, 2.2, 3.0, 3.8]
    for sc in scales:
        roi = roi0
        if sc != 1.0:
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        for r in rotate_variants(roi):
            for v in _variants_for_qr(r):
                s = _try_decode(det, v)
                if s:
                    return s, debug_qr

    # 3) ROI más grande por si viene desplazado
    top = img_bgr[0:1700, 0:1700].copy()
    for v in _variants_for_qr(top):
        s = _try_decode(det, v)
        if s:
            return s, debug_qr

    return None, debug_qr


def parsear_codigo_qr(codigo):
    """
    Formato esperado (mínimo):
      id_examen|id_alumno|num_preguntas
    Opcional:
      |pagina   (1 o 2)
    """
    if not codigo:
        return None
    partes = [p.strip() for p in codigo.split("|")]
    if len(partes) < 3:
        return None
    try:
        id_examen = int(partes[0])
        id_alumno = int(partes[1])
        num_preguntas = int(partes[2])
        pagina = 1
        if len(partes) >= 4 and partes[3].isdigit():
            pagina = int(partes[3])
        pagina = 1 if pagina not in (1, 2) else pagina
        return id_examen, id_alumno, num_preguntas, pagina
    except Exception:
        return None


# =========================
# Binarización para tinta (móvil + escáner)
# =========================
def binarizar_tinta_pro(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
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


# =========================
# Filas de esta página según QR
# =========================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0

    # página 2
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA

    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# =========================
# Detectar círculos impresos para construir rejilla real (clave)
# =========================
def detectar_circulos_burbujas(img_a4_bgr, filas_objetivo):
    """
    Devuelve:
      centers: lista de (x,y,r)
    Usa HoughCircles sobre un ROI aproximado.
    """
    y0, x0, y1, x1 = BUBBLES_SEARCH_ROI
    roi = img_a4_bgr[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Parámetros pensados para tus círculos (bastante grandes)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=120,
        param2=35,
        minRadius=16,
        maxRadius=45
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)

    # Convertir coords a A4
    out = []
    for (cx, cy, r) in circles:
        X = cx + x0
        Y = cy + y0
        out.append((X, Y, r))

    # Filtrar por zona “razonable”: cerca del bloque de respuestas
    # (quita falsos positivos del marco)
    filtered = []
    for (X, Y, r) in out:
        if 350 < X < 1600 and 250 < Y < 2900:
            filtered.append((X, Y, r))

    return filtered


def kmeans_1d(values, k):
    """
    KMeans 1D usando OpenCV. values -> lista/array shape (N,)
    devuelve centros ordenados.
    """
    v = np.array(values, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(v, k, None, criteria, 10, flags)
    centers = centers.flatten()
    centers_sorted = np.sort(centers)
    return centers_sorted


def construir_rejilla(circulos, filas_objetivo):
    """
    A partir de círculos detectados:
      - agrupa X en 4 columnas (A-D)
      - agrupa Y en filas_objetivo filas
    Devuelve:
      col_centers (4), row_centers (filas_objetivo), radio_medio
    """
    if len(circulos) < 12:
        return None

    xs = [c[0] for c in circulos]
    ys = [c[1] for c in circulos]
    rs = [c[2] for c in circulos]

    # 4 columnas fijas
    col_centers = kmeans_1d(xs, 4)

    # filas según QR
    filas = max(1, int(filas_objetivo))
    row_centers = kmeans_1d(ys, filas)

    radio = float(np.median(rs))
    return col_centers, row_centers, radio


# =========================
# Medir tinta dentro de cada círculo (con máscara circular)
# =========================
def score_circulo(bin_img, cx, cy, r):
    """
    bin_img: A4 binaria (tinta=255)
    """
    r_in = int(max(6, r * RADIO_INTERIOR_RATIO))

    y0 = max(0, cy - r_in)
    y1 = min(bin_img.shape[0], cy + r_in)
    x0 = max(0, cx - r_in)
    x1 = min(bin_img.shape[1], cx + r_in)

    patch = bin_img[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    # máscara circular
    hh, ww = patch.shape
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(mask, (ww // 2, hh // 2), r_in, 255, -1)

    ink = cv2.countNonZero(cv2.bitwise_and(patch, patch, mask=mask))
    area = cv2.countNonZero(mask)
    if area <= 0:
        return 0.0
    return ink / float(area)


def detectar_respuestas_por_circulos(th_a4, debug_a4, col_centers, row_centers, radio, filas):
    """
    th_a4: binaria A4 tinta=255
    debug_a4: A4 color para pintar
    """
    respuestas = []

    col_centers = list(col_centers)
    row_centers = list(row_centers)

    # ordenar columnas izquierda->derecha (A,B,C,D)
    col_centers.sort()
    # ordenar filas arriba->abajo
    row_centers.sort()

    for i in range(filas):
        cy = int(row_centers[i])

        scores = []
        for j in range(4):
            cx = int(col_centers[j])
            s = score_circulo(th_a4, cx, cy, int(radio))
            scores.append(s)

            # debug: dibuja círculo y score
            cv2.circle(debug_a4, (cx, cy), int(radio), (0, 255, 0), 2)

        max_s = float(np.max(scores))
        idx = int(np.argmax(scores))
        second = float(np.sort(scores)[-2])

        if max_s < UMBRAL_VACIO:
            resp = ""
        elif (second > UMBRAL_DOBLE_ABS) and (second > max_s * UMBRAL_DOBLE_RATIO):
            resp = "X"
        else:
            resp = OPCIONES[idx]

        respuestas.append(resp)

        # resaltar elegida
        if resp and resp != "X":
            cx = int(col_centers[idx])
            cv2.circle(debug_a4, (cx, cy), int(radio) + 2, (0, 0, 255), 3)

        # etiqueta fila
        cv2.putText(
            debug_a4,
            f"{i+1}:{resp or '-'}",
            (int(col_centers[0]) - 140, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if resp == "X" else (0, 0, 0),
            2
        )

    return respuestas, debug_a4


# =========================
# Pipeline principal
# =========================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 1) QR antes de normalizar (a veces sale mejor)
    qr0, dbg_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(qr0) if qr0 else None

    # 2) Normalizar A4
    img_a4 = normalizar_a4_con_marcas(img)

    # 3) QR después de normalizar (si no lo teníamos)
    qr, dbg_qr = qr0, dbg_qr0
    parsed = parsed0
    if not parsed:
        qr, dbg_qr = leer_qr_robusto(img_a4)
        parsed = parsear_codigo_qr(qr) if qr else None

    if not parsed:
        # devolvemos recorte de QR para depurar
        return {
            "ok": False,
            "error": "QR no detectado",
            "debug_qr": dbg_qr
        }

    id_examen, id_alumno, num_preguntas, pagina = parsed

    # 4) filas de esta página
    filas, offset = filas_a_leer(num_preguntas, pagina)
    if filas <= 0:
        return {
            "ok": False,
            "error": "Esta página no tiene preguntas según el QR",
            "codigo": qr,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": dbg_qr
        }

    # 5) binarizar tinta en A4
    th_a4 = binarizar_tinta_pro(img_a4)

    # 6) detectar círculos impresos y construir rejilla real
    circulos = detectar_circulos_burbujas(img_a4, filas)
    if len(circulos) < max(8, filas * 2):  # mínima evidencia
        return {
            "ok": False,
            "error": "No se detectaron suficientes burbujas (círculos) para construir la rejilla",
            "codigo": qr,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": dbg_qr
        }

    rej = construir_rejilla(circulos, filas)
    if rej is None:
        return {
            "ok": False,
            "error": "No se pudo construir la rejilla (kmeans)",
            "codigo": qr,
            "debug_qr": dbg_qr
        }

    col_centers, row_centers, radio = rej

    debug_a4 = img_a4.copy()

    # 7) detectar respuestas usando círculos reales
    resp_lista, debug_a4 = detectar_respuestas_por_circulos(
        th_a4, debug_a4, col_centers, row_centers, radio, filas
    )

    # 8) salida dict 1..60 según offset
    respuestas = {}
    for i, r in enumerate(resp_lista, start=1):
        respuestas[str(offset + i)] = r

    return {
        "ok": True,
        "codigo": qr,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "num_preguntas": num_preguntas,
        "pagina": pagina,
        "respuestas": respuestas,
        "debug_image": b64jpg(debug_a4, 85),
        "debug_qr": dbg_qr
    }


# =========================
# Endpoints
# =========================
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
