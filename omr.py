from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import re

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

# Zona OMR amplia y alta para no perder la pregunta 1
OMR_REGION = {
    "y0": 360,
    "y1": 3300,
    "x0": 450,
    "x1": 1900
}

MAX_FILAS_POR_HOJA = 30

# Umbrales
UMBRAL_VACIO = 0.080
UMBRAL_DOBLE_RATIO = 0.86
UMBRAL_DOBLE_ABS = 0.060

# Geometría de evaluación circular
R_INNER = 0.42       # radio interior real a medir
R_RING_IN = 0.62     # inicio anillo borde
R_RING_OUT = 0.95    # fin anillo borde

# penalización del borde negro impreso
BORDE_PENALIZACION = 0.55

# ============================================================
# UTIL
# ============================================================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


def safe_crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


# ============================================================
# NORMALIZAR A4 CON MARCAS
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
        if bw <= 0 or bh <= 0:
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
        best_i, best_d = None, None
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
    return cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))


# ============================================================
# QR ROBUSTO
# ============================================================
def _try_decode(det, img_bgr):
    try:
        ok, decoded_info, _, _ = det.detectAndDecodeMulti(img_bgr)
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
    out = [img_bgr]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    out.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
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

    rotations = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    for im in rotations:
        for v in _variants(im):
            s = _try_decode(det, v)
            if s:
                return s, None

    qr_roi = safe_crop(img_bgr, 0, 0, 1400, 1400)
    debug_qr = b64jpg(qr_roi, 90) if qr_roi is not None else None
    if qr_roi is None:
        return None, None

    scales = [1.0, 1.8, 2.6, 3.4, 4.0]
    for sc in scales:
        roi = qr_roi
        if sc != 1.0:
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        rots = [
            roi,
            cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(roi, cv2.ROTATE_180),
            cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        for r in rots:
            for v in _variants(r):
                s = _try_decode(det, v)
                if s:
                    return s, debug_qr

    big = safe_crop(img_bgr, 0, 0, 1700, 1700)
    if big is not None:
        for v in _variants(big):
            s = _try_decode(det, v)
            if s:
                return s, debug_qr

    return None, debug_qr


def parsear_codigo_qr(codigo):
    """
    Formato real:
      id_examen|id_alumno|fecha|num_preguntas|pagina(opcional)
    Ej:
      261|276|2026-02-16|20
      261|276|2026-02-16|60|2
    """
    if not codigo:
        return None

    if "|" in codigo:
        partes = [p.strip() for p in codigo.split("|")]
        if len(partes) >= 4:
            try:
                id_examen = int(partes[0])
                id_alumno = int(partes[1])
                fecha = partes[2]
                num_preg = int(partes[3])
                pagina = int(partes[4]) if len(partes) >= 5 and str(partes[4]).isdigit() else 1
                pagina = 1 if pagina not in (1, 2) else pagina
                return id_examen, id_alumno, fecha, num_preg, pagina
            except:
                pass

    nums = re.findall(r"\d+", codigo)
    if len(nums) < 6:
        return None

    try:
        id_examen = int(nums[0])
        id_alumno = int(nums[1])
        fecha = f"{nums[2]}-{nums[3].zfill(2)}-{nums[4].zfill(2)}"
        num_preg = int(nums[5])
        pagina = int(nums[6]) if len(nums) >= 7 else 1
        pagina = 1 if pagina not in (1, 2) else pagina
        return id_examen, id_alumno, fecha, num_preg, pagina
    except:
        return None


# ============================================================
# BINARIZACIÓN TINTA
# ============================================================
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
# DETECCIÓN REAL DE CÍRCULOS
# ============================================================
def detectar_circulos(zona_gray):
    g = cv2.medianBlur(zona_gray, 5)

    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=34,
        param1=120,
        param2=26,
        minRadius=14,
        maxRadius=34
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")
    out = []
    h, w = zona_gray.shape[:2]

    for (x, y, r) in circles:
        if 0 <= x < w and 0 <= y < h:
            out.append((int(x), int(y), int(r)))

    return out


def agrupar_filas(circulos, filas_esperadas):
    if not circulos:
        return []

    idx_sorted = sorted(range(len(circulos)), key=lambda i: circulos[i][1])

    ys = [circulos[i][1] for i in idx_sorted]
    diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
    diffs = [d for d in diffs if d > 2]

    if not diffs:
        return []

    thr = max(16.0, float(np.median(diffs)) * 0.65)

    filas = []
    actual = [idx_sorted[0]]
    y_ref = circulos[idx_sorted[0]][1]

    for idx in idx_sorted[1:]:
        y = circulos[idx][1]
        if abs(y - y_ref) <= thr:
            actual.append(idx)
            y_ref = 0.7 * y_ref + 0.3 * y
        else:
            filas.append(actual)
            actual = [idx]
            y_ref = y
    filas.append(actual)

    filas.sort(key=lambda fila: np.mean([circulos[i][1] for i in fila]))
    if len(filas) > filas_esperadas:
        filas = filas[:filas_esperadas]

    return filas


def cluster_columnas_x(circulos):
    xs = np.array([[c[0]] for c in circulos], dtype=np.float32)
    if len(xs) < 8:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(xs, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = sorted([float(c[0]) for c in centers])
    return centers


def score_circulo(gray, cx, cy, r):
    """
    Score bueno:
      oscuridad interior - penalización por borde
    Así no confunde borde negro con relleno.
    """
    h, w = gray.shape[:2]

    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    hh, ww = roi.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    dx = xx - (cx - x0)
    dy = yy - (cy - y0)
    dist = np.sqrt(dx * dx + dy * dy)

    inner_mask = dist <= (r * R_INNER)
    ring_mask = (dist >= (r * R_RING_IN)) & (dist <= (r * R_RING_OUT))

    if inner_mask.sum() == 0 or ring_mask.sum() == 0:
        return 0.0

    # oscuridad = 1 - brillo_normalizado
    roi_f = roi.astype(np.float32) / 255.0
    dark = 1.0 - roi_f

    inner_dark = float(np.mean(dark[inner_mask]))
    ring_dark = float(np.mean(dark[ring_mask]))

    score = inner_dark - BORDE_PENALIZACION * ring_dark
    return max(0.0, score)


def detectar_respuestas_por_circulos(img_a4, filas, debug=True):
    debug_a4 = img_a4.copy() if debug else None

    zona_color = safe_crop(img_a4, OMR_REGION["x0"], OMR_REGION["y0"], OMR_REGION["x1"], OMR_REGION["y1"])
    if zona_color is None:
        return [], debug_a4

    zona_gray = cv2.cvtColor(zona_color, cv2.COLOR_BGR2GRAY)

    if debug_a4 is not None:
        cv2.rectangle(
            debug_a4,
            (OMR_REGION["x0"], OMR_REGION["y0"]),
            (OMR_REGION["x1"], OMR_REGION["y1"]),
            (255, 0, 0),
            3
        )

    circles = detectar_circulos(zona_gray)

    if debug_a4 is not None:
        for (x, y, r) in circles:
            cv2.circle(debug_a4, (OMR_REGION["x0"] + x, OMR_REGION["y0"] + y), r, (0, 180, 255), 2)

    if len(circles) < 20:
        return [], debug_a4

    filas_groups = agrupar_filas(circles, filas)
    if len(filas_groups) < max(5, int(filas * 0.5)):
        return [], debug_a4

    col_centers = cluster_columnas_x(circles)
    if not col_centers or len(col_centers) != 4:
        return [], debug_a4

    respuestas = []

    for row_i in range(filas):
        if row_i >= len(filas_groups):
            respuestas.append("")
            continue

        idxs = filas_groups[row_i]
        scores = {c: 0.0 for c in OPCIONES}
        coords = {}

        for ci, letra in enumerate(OPCIONES):
            target_x = col_centers[ci]

            best = None
            best_dx = None
            for idx in idxs:
                x, y, r = circles[idx]
                dx = abs(x - target_x)
                if best_dx is None or dx < best_dx:
                    best_dx = dx
                    best = (x, y, r)

            if best is None:
                scores[letra] = 0.0
                coords[letra] = None
                continue

            x, y, r = best
            coords[letra] = (x, y, r)
            scores[letra] = score_circulo(zona_gray, x, y, r)

        orden = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_letter, best_val = orden[0]
        second_val = orden[1][1]

        if best_val < UMBRAL_VACIO:
            resp = ""
        elif (second_val > UMBRAL_DOBLE_ABS) and (second_val > best_val * UMBRAL_DOBLE_RATIO):
            resp = "X"
        else:
            resp = best_letter

        respuestas.append(resp)

        if debug_a4 is not None:
            for letra in OPCIONES:
                c = coords.get(letra)
                if c is None:
                    continue
                x, y, r = c
                X = OMR_REGION["x0"] + x
                Y = OMR_REGION["y0"] + y
                cv2.circle(debug_a4, (X, Y), r, (0, 255, 0), 2)

            if resp in coords and coords[resp] is not None and resp != "X":
                x, y, r = coords[resp]
                X = OMR_REGION["x0"] + x
                Y = OMR_REGION["y0"] + y
                cv2.circle(debug_a4, (X, Y), r, (0, 0, 255), 3)

            y_mean = int(np.mean([circulos[i][1] for i in idxs]))
            cv2.putText(
                debug_a4,
                f"{row_i+1}:{resp or '-'}",
                (OMR_REGION["x0"] - 170, OMR_REGION["y0"] + y_mean + 8),
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

    # 1) QR antes de normalizar
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 2) normalizar A4
    img_a4 = normalizar_a4_con_marcas(img)

    # 3) QR después si falló
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

    id_examen, id_alumno, fecha, num_preguntas, pagina = parsed

    filas, offset = filas_a_leer(num_preguntas, pagina)
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

    respuestas_lista, debug_a4 = detectar_respuestas_por_circulos(img_a4, filas, debug=True)
    if not respuestas_lista:
        return {
            "ok": False,
            "error": "No se pudieron detectar bien las burbujas OMR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha": fecha,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr,
            "debug_image": b64jpg(debug_a4, 85) if debug_a4 is not None else None
        }

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
        "debug_qr": debug_qr,
        "debug_image": b64jpg(debug_a4, 85) if debug_a4 is not None else None
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
