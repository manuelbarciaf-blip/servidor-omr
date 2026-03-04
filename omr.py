from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# =========================
# CONFIG
# =========================
A4_W, A4_H = 2480, 3508  # A4 ~300dpi tras normalizar
OPCIONES = ["A", "B", "C", "D"]

MAX_FILAS_POR_HOJA = 30   # tu hoja mete hasta 30 en una página
UMBRAL_VACIO = 0.055      # ratio tinta mínimo para considerar marcada
UMBRAL_DOBLE_RATIO = 0.78 # si la 2ª está muy cerca de la 1ª -> doble
UMBRAL_DOBLE_ABS = 0.040  # y además supera un mínimo absoluto

# ROI interior para ignorar borde del círculo
INNER_PAD = 0.55  # 0.55 = usar ~55% del radio hacia dentro (zona sólida)

# ROI AMPLIA donde buscamos burbujas (en A4 normalizado)
# (NO es “zona exacta”, es una zona grande para encontrar círculos)
OMR_ROI_WIDE = {"x0": 450, "y0": 400, "x1": 1700, "y1": 3150}

# ROI amplia donde suele estar el QR (en A4 normalizado)
QR_ROI_WIDE = {"x0": 0, "y0": 0, "x1": 1400, "y1": 1400}


# =========================
# Utils
# =========================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


def clamp(v, a, b):
    return max(a, min(b, v))


# =========================
# Normalizar A4 con marcas negras
# - Mejorado: elegimos mejor marca por cuadrante
# =========================
def normalizar_a4_con_marcas(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    midx, midy = w / 2, h / 2

    # guardamos mejor candidato por cuadrante (por área)
    best = {"tl": None, "tr": None, "bl": None, "br": None}

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh else 0
        if not (0.60 < aspect < 1.40):
            continue

        cx = x + bw / 2
        cy = y + bh / 2

        if cx < midx and cy < midy:
            key = "tl"
        elif cx >= midx and cy < midy:
            key = "tr"
        elif cx < midx and cy >= midy:
            key = "bl"
        else:
            key = "br"

        if best[key] is None or area > best[key]["area"]:
            best[key] = {"pt": (cx, cy), "area": area}

    if any(best[k] is None for k in best):
        # fallback simple si faltan marcas
        return cv2.resize(img_bgr, (A4_W, A4_H))

    tl = np.array(best["tl"]["pt"], dtype=np.float32)
    tr = np.array(best["tr"]["pt"], dtype=np.float32)
    br = np.array(best["br"]["pt"], dtype=np.float32)
    bl = np.array(best["bl"]["pt"], dtype=np.float32)

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))

    return warped


# =========================
# QR robusto (sin pyzbar)
# - Multi-intento + preprocesados + escalas + rotaciones
# =========================
def _try_decode(det, img_bgr):
    # multi si existe
    try:
        ok, infos, pts, _ = det.detectAndDecodeMulti(img_bgr)
        if ok and infos:
            for s in infos:
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

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    out.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # sharpen
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sh = cv2.filter2D(gray, -1, k)
    out.append(cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))

    # otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # adaptive
    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    out.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - ad, cv2.COLOR_GRAY2BGR))

    return out


def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # 1) imagen completa
    s = _try_decode(det, img_bgr)
    if s:
        return s, None

    # 2) ROI superior izq grande
    h, w = img_bgr.shape[:2]
    x0, y0 = 0, 0
    x1, y1 = int(w * 0.55), int(h * 0.55)
    roi = img_bgr[y0:y1, x0:x1].copy()

    debug_roi = roi.copy()

    for scale in [1.0, 1.8, 2.5, 3.2]:
        r = roi
        if scale != 1.0:
            r = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        rots = [
            r,
            cv2.rotate(r, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(r, cv2.ROTATE_180),
            cv2.rotate(r, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        for rr in rots:
            for v in _variants_for_qr(rr):
                s = _try_decode(det, v)
                if s:
                    return s, b64jpg(debug_roi)

    return None, b64jpg(debug_roi)


def parsear_codigo_qr(codigo):
    """
    Formato recomendado:
      id_examen|id_alumno|num_preguntas|pagina
    pagina opcional (1 o 2)
    """
    partes = (codigo or "").split("|")
    if len(partes) < 3:
        return None
    try:
        id_examen = int(partes[0])
        id_alumno = int(partes[1])
        num_preg = int(partes[2])
        pagina = int(partes[3]) if len(partes) >= 4 and str(partes[3]).strip().isdigit() else 1
        pagina = 1 if pagina not in (1, 2) else pagina
        num_preg = clamp(num_preg, 1, 60)
        return id_examen, id_alumno, num_preg, pagina
    except Exception:
        return None


# =========================
# Binarización tinta (móvil+escáner)
# =========================
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


# =========================
# Filas según QR + página
# =========================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# =========================
# Detectar círculos (grid real)
# =========================
def detectar_circulos_grid(img_a4_bgr, debug_a4):
    """
    Devuelve lista de círculos detectados (x,y,r) en coordenadas A4,
    filtrados dentro de la ROI OMR amplia.
    """
    x0, y0, x1, y1 = OMR_ROI_WIDE["x0"], OMR_ROI_WIDE["y0"], OMR_ROI_WIDE["x1"], OMR_ROI_WIDE["y1"]
    roi = img_a4_bgr[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=120,
        param2=28,
        minRadius=14,
        maxRadius=38,
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)

    out = []
    for (x, y, r) in circles:
        ax = x0 + x
        ay = y0 + y
        out.append((ax, ay, r))

    # debug: dibujar todos
    for (ax, ay, r) in out:
        cv2.circle(debug_a4, (ax, ay), r, (0, 255, 0), 2)

    return out


def agrupar_columnas(circulos):
    """
    Agrupa por X en 4 columnas usando clustering simple por percentiles.
    Devuelve 4 centros X ordenados.
    """
    xs = sorted([c[0] for c in circulos])
    if len(xs) < 12:
        return None

    # aproximación robusta: usar quantiles para encontrar 4 bandas
    q = np.quantile(xs, [0.15, 0.40, 0.65, 0.90])
    cols = []
    for target in q:
        near = [x for x in xs if abs(x - target) < 90]
        if near:
            cols.append(int(np.median(near)))

    # si sale menos de 4, fallback por kmeans manual (muy simple)
    cols = sorted(list(set(cols)))
    if len(cols) < 4:
        # fallback: dividir en 4 grupos por orden
        xs_np = np.array(xs)
        groups = np.array_split(xs_np, 4)
        cols = [int(np.median(g)) for g in groups]

    # elegir 4 y ordenar
    cols = sorted(cols)[:4]
    return cols if len(cols) == 4 else None


def agrupar_filas(circulos):
    """
    Agrupa por Y en filas (centros). Devuelve lista de Y ordenados.
    """
    ys = sorted([c[1] for c in circulos])
    if len(ys) < 12:
        return None

    filas = []
    tol = 28  # tolerancia para agrupar centros en la misma fila

    for y in ys:
        if not filas or abs(y - filas[-1]) > tol:
            filas.append(y)
        else:
            # “promediar” para estabilizar
            filas[-1] = int((filas[-1] + y) / 2)

    return filas


def score_circulo(th_bin, cx, cy, r):
    """
    th_bin: binaria invertida (tinta=blanco/255)
    score = ratio pixeles blancos en un disco interior
    """
    h, w = th_bin.shape[:2]
    rr = int(r * INNER_PAD)
    rr = max(4, rr)

    x0 = clamp(cx - rr, 0, w - 1)
    x1 = clamp(cx + rr, 0, w - 1)
    y0 = clamp(cy - rr, 0, h - 1)
    y1 = clamp(cy + rr, 0, h - 1)

    roi = th_bin[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    # máscara circular
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), rr, 255, -1)

    ink = cv2.bitwise_and(roi, roi, mask=mask)
    score = cv2.countNonZero(ink) / float(cv2.countNonZero(mask) + 1e-6)
    return float(score)


def detectar_respuestas_por_circulos(img_a4_bgr, th_a4, filas_real, offset_preg, debug_a4):
    """
    Detecta respuestas encontrando los círculos y construyendo la rejilla real.
    """
    circulos = detectar_circulos_grid(img_a4_bgr, debug_a4)
    if not circulos:
        return None, "No se detectaron círculos OMR (HoughCircles)", debug_a4

    cols_x = agrupar_columnas(circulos)
    filas_y = agrupar_filas(circulos)

    if cols_x is None or filas_y is None:
        return None, "No se pudo construir la rejilla (columnas/filas)", debug_a4

    # Nos quedamos con filas de la zona OMR (más arriba a más abajo)
    filas_y = sorted(filas_y)

    # Si hay más filas detectadas que MAX, nos quedamos con las más “centrales” típicas
    # pero como tu hoja suele tener 20 o 30, esto ayuda a no confundirse con otros círculos.
    if len(filas_y) > MAX_FILAS_POR_HOJA:
        filas_y = filas_y[:MAX_FILAS_POR_HOJA]

    # Solo leer las filas que tocan según QR
    filas_a_usar = filas_y[:filas_real]
    if len(filas_a_usar) < filas_real:
        # si detectó menos filas de las que pide el QR
        filas_real = len(filas_a_usar)

    respuestas = {}
    for i in range(filas_real):
        y = filas_a_usar[i]
        scores = []

        # estimar radio medio de la fila (tomando círculos cercanos)
        rs = [c[2] for c in circulos if abs(c[1] - y) < 35]
        r_med = int(np.median(rs)) if rs else 22

        for j, x in enumerate(cols_x):
            s = score_circulo(th_a4, x, y, r_med)
            scores.append(s)

        max_s = max(scores)
        idx = int(np.argmax(scores))
        sorted_s = sorted(scores, reverse=True)
        second = sorted_s[1] if len(sorted_s) > 1 else 0.0

        if max_s < UMBRAL_VACIO:
            resp = ""
        elif (second > UMBRAL_DOBLE_ABS) and (second > max_s * UMBRAL_DOBLE_RATIO):
            resp = "X"
        else:
            resp = OPCIONES[idx]

        num_global = offset_preg + (i + 1)
        respuestas[str(num_global)] = resp

        # debug: remarcar la elegida
        for j, x in enumerate(cols_x):
            color = (0, 200, 0)
            if resp and resp != "X" and OPCIONES[j] == resp:
                color = (0, 0, 255)
            cv2.circle(debug_a4, (int(x), int(y)), int(r_med), color, 3)

        cv2.putText(
            debug_a4,
            f"{num_global}:{resp or '-'}",
            (int(cols_x[0] - 120), int(y + 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if resp == "X" else (0, 0, 0),
            2
        )

    return respuestas, None, debug_a4


# =========================
# Pipeline principal
# =========================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) intentar QR ANTES de normalizar (a veces mejor)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) normalizar con marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR tras normalizar si hizo falta
    codigo, debug_qr = codigo0, debug_qr0
    parsed = parsed0
    if not parsed:
        codigo, debug_qr = leer_qr_robusto(img_a4)
        parsed = parsear_codigo_qr(codigo) if codigo else None

    if not parsed:
        # devolver debug del ROI QR para ajustar
        # además añadimos un recorte A4 de la zona QR típica
        qr_crop = img_a4[QR_ROI_WIDE["y0"]:QR_ROI_WIDE["y1"], QR_ROI_WIDE["x0"]:QR_ROI_WIDE["x1"]].copy()
        return {
            "ok": False,
            "error": "QR no detectado",
            "debug_qr": debug_qr,
            "debug_qr_a4": b64jpg(qr_crop)
        }

    id_examen, id_alumno, num_preguntas, pagina = parsed

    # 3) binarizar tinta (sobre A4)
    th_a4 = binarizar_tinta_pro(img_a4)

    # 4) filas a leer según QR
    filas_real, offset = filas_a_leer(num_preguntas, pagina)
    if filas_real <= 0:
        return {
            "ok": False,
            "error": "Esta página no tiene preguntas según el QR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr,
        }

    # 5) detectar respuestas por círculos (sin recorte pequeño)
    debug_a4 = img_a4.copy()
    respuestas, err_grid, debug_a4 = detectar_respuestas_por_circulos(
        img_a4_bgr=img_a4,
        th_a4=th_a4,
        filas_real=filas_real,
        offset_preg=offset,
        debug_a4=debug_a4
    )

    if respuestas is None:
        # devolver debug de la zona OMR amplia
        roi = img_a4[OMR_ROI_WIDE["y0"]:OMR_ROI_WIDE["y1"], OMR_ROI_WIDE["x0"]:OMR_ROI_WIDE["x1"]].copy()
        return {
            "ok": False,
            "error": err_grid or "No se pudo detectar OMR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr,
            "debug_omr_roi": b64jpg(roi),
            "debug_image": b64jpg(debug_a4)
        }

    return {
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "num_preguntas": num_preguntas,
        "pagina": pagina,
        "respuestas": respuestas,
        "debug_qr": debug_qr,
        "debug_image": b64jpg(debug_a4, 85),
    }


# =========================
# Endpoint
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
