from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import re

app = Flask(__name__)

# ============================================================
# CONFIG ✅ (Ajustada a tu hoja)
# ============================================================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

# Zona OMR grande (para encontrar círculos). La ajustamos más ARRIBA para incluir la 1.
# Si tu plantilla cambia, retoca estos 4 números.
OMR_REGION = {
    "y0": 360,     # ⬅️ subimos para coger la pregunta 1
    "y1": 3300,    # ⬅️ bajamos para coger hasta la 30
    "x0": 450,     # ⬅️ ampliamos izquierda
    "x1": 1900     # ⬅️ ampliamos derecha
}

MAX_FILAS_POR_HOJA = 30

# Umbrales lectura burbujas (tinta azul/negra)
UMBRAL_VACIO = 0.045          # si está demasiado alto -> lee blancos
UMBRAL_DOBLE_RATIO = 0.88     # doble si 2ª se acerca a la 1ª
UMBRAL_DOBLE_ABS = 0.038      # y además supera un mínimo absoluto

# ROI interior para evitar contar borde impreso
INNER_PAD = 0.30  # 0.25–0.35 suele ir bien

# ============================================================
# UTIL
# ============================================================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


def _safe_crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


# ============================================================
# 1) NORMALIZAR A4 con marcas negras (robusto)
# ============================================================
def normalizar_a4_con_marcas(img_bgr):
    """
    Detecta 4 marcas negras de esquina y aplica perspectiva a A4.
    Si falla, hace resize a A4.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # negro -> blanco
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

    # prioridad por área (marcas grandes)
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
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))
    return warped


# ============================================================
# 2) QR ROBUSTO (MUCHOS INTENTOS)
# ============================================================
def _try_decode(det, img_bgr):
    # multi si está disponible
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
    out = [img_bgr]
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

    # Otsu
    _, otsu = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR))

    # Adaptive
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

    # 0) Intento completo con variantes y rotaciones
    rotations = [
        ("0", img_bgr),
        ("90", cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)),
        ("180", cv2.rotate(img_bgr, cv2.ROTATE_180)),
        ("270", cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    for _, im in rotations:
        for v in _variants(im):
            s = _try_decode(det, v)
            if s:
                return s, None

    # 1) ROI fijo arriba izquierda
    # (en tu hoja el QR SIEMPRE está ahí)
    qr_roi = _safe_crop(img_bgr, 0, 0, 1400, 1400)
    debug_qr = b64jpg(qr_roi, 90) if qr_roi is not None else None
    if qr_roi is None:
        return None, None

    scales = [1.0, 1.8, 2.6, 3.4, 4.0]
    for sc in scales:
        roi = qr_roi
        if sc != 1.0:
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        # rotaciones del ROI
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

    # 2) ROI más grande (por si el QR está más desplazado)
    big = _safe_crop(img_bgr, 0, 0, 1700, 1700)
    if big is not None:
        for v in _variants(big):
            s = _try_decode(det, v)
            if s:
                return s, debug_qr

    return None, debug_qr


def parsear_codigo_qr(codigo):
    """
    ✅ Formato real de tu QR:
      id_examen|id_alumno|fecha|num_preguntas|pagina(opcional)

    Ej: 261|276|2026-02-16|20
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
                pagina = int(partes[4]) if len(partes) >= 5 and str(partes[4]).strip().isdigit() else 1
                pagina = 1 if pagina not in (1, 2) else pagina
                return id_examen, id_alumno, fecha, num_preg, pagina
            except:
                pass

    # fallback regex (por si metes texto)
    nums = re.findall(r"\d+", codigo)
    if len(nums) < 3:
        return None
    try:
        id_examen = int(nums[0])
        id_alumno = int(nums[1])
        num_preg = int(nums[-1])  # el último suele ser num preguntas
        # fecha no se puede reconstruir fiable aquí => dejamos None
        pagina = 1
        return id_examen, id_alumno, None, num_preg, pagina
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
# 5) DETECCIÓN REAL DE CÍRCULOS + AGRUPACIÓN (SIN REJILLA FIJA)
# ============================================================
def detectar_circulos(zona_gray):
    """
    Devuelve lista de círculos (x, y, r) en coordenadas de la zona.
    """
    g = cv2.medianBlur(zona_gray, 5)

    # HoughCircles: ajustado a tus círculos
    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=38,
        param1=120,
        param2=30,
        minRadius=16,
        maxRadius=45
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")
    # filtrar por zona útil (evitar falsos en bordes)
    h, w = zona_gray.shape[:2]
    out = []
    for (x, y, r) in circles:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        out.append((int(x), int(y), int(r)))
    return out


def agrupar_filas(circulos, filas_esperadas):
    """
    Agrupa círculos por filas usando distancias en Y (sin asumir rejilla fija).
    Devuelve: lista de filas, cada fila es lista de indices de círculos.
    """
    if not circulos:
        return []

    ys = sorted([c[1] for c in circulos])
    if len(ys) < 8:
        return []

    diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
    diffs = [d for d in diffs if d > 1]
    if not diffs:
        return []

    med = float(np.median(diffs))
    thr = max(14.0, med * 0.60)

    # ordenar por Y
    idx_sorted = sorted(range(len(circulos)), key=lambda i: circulos[i][1])

    filas = []
    actual = [idx_sorted[0]]
    y_ref = circulos[idx_sorted[0]][1]

    for idx in idx_sorted[1:]:
        y = circulos[idx][1]
        if abs(y - y_ref) <= thr:
            actual.append(idx)
            y_ref = (y_ref * 0.7 + y * 0.3)
        else:
            filas.append(actual)
            actual = [idx]
            y_ref = y
    filas.append(actual)

    # Orden por y medio
    filas.sort(key=lambda fila: np.mean([circulos[i][1] for i in fila]))

    # Nos quedamos con las primeras filas_esperadas (la hoja empieza arriba)
    if len(filas) > filas_esperadas:
        filas = filas[:filas_esperadas]

    return filas


def cluster_columnas_x(circulos):
    """
    Encuentra 4 centros de columna en X mediante kmeans.
    """
    xs = np.array([[c[0]] for c in circulos], dtype=np.float32)
    if len(xs) < 8:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    K = 4
    ret, labels, centers = cv2.kmeans(xs, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = sorted([float(c[0]) for c in centers])
    return centers


def score_circulo(mask_bin, cx, cy, r):
    """
    Calcula densidad de tinta dentro del círculo (en la máscara binaria).
    """
    h, w = mask_bin.shape[:2]

    pad = int(r * INNER_PAD)
    rr = max(8, int(r * 0.72))

    x0 = max(0, cx - rr)
    x1 = min(w, cx + rr)
    y0 = max(0, cy - rr)
    y1 = min(h, cy + rr)

    roi = mask_bin[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    # máscara circular interior
    hh, ww = roi.shape[:2]
    yy, xx = np.ogrid[:hh, :ww]
    cxx = cx - x0
    cyy = cy - y0
    rad = max(6, rr - pad)
    circle_mask = ((xx - cxx) ** 2 + (yy - cyy) ** 2) <= (rad ** 2)

    inside = roi[circle_mask]
    if inside.size == 0:
        return 0.0

    return float(cv2.countNonZero(inside)) / float(inside.size)


def detectar_respuestas_por_circulos(img_a4, th_bin, filas, debug=True):
    """
    Detecta respuestas usando círculos reales. Devuelve:
      respuestas_lista, debug_a4
    """
    debug_a4 = img_a4.copy() if debug else None

    # recortar zona OMR
    zona_color = _safe_crop(img_a4, OMR_REGION["x0"], OMR_REGION["y0"], OMR_REGION["x1"], OMR_REGION["y1"])
    zona_bin = _safe_crop(th_bin, OMR_REGION["x0"], OMR_REGION["y0"], OMR_REGION["x1"], OMR_REGION["y1"])

    if zona_color is None or zona_bin is None:
        return [], debug_a4

    zona_gray = cv2.cvtColor(zona_color, cv2.COLOR_BGR2GRAY)

    # dibujar rectángulo OMR
    if debug_a4 is not None:
        cv2.rectangle(
            debug_a4,
            (OMR_REGION["x0"], OMR_REGION["y0"]),
            (OMR_REGION["x1"], OMR_REGION["y1"]),
            (255, 0, 0),
            3
        )

    circles = detectar_circulos(zona_gray)

    # Debug: círculos detectados
    if debug_a4 is not None:
        for (x, y, r) in circles:
            cv2.circle(debug_a4, (OMR_REGION["x0"] + x, OMR_REGION["y0"] + y), r, (0, 180, 255), 2)

    if len(circles) < 20:
        # muy pocos círculos => no fiable
        return [], debug_a4

    # agrupar filas
    filas_groups = agrupar_filas(circles, filas)
    if len(filas_groups) < max(5, int(filas * 0.5)):
        return [], debug_a4

    # centros de columnas
    col_centers = cluster_columnas_x(circles)
    if not col_centers or len(col_centers) != 4:
        return [], debug_a4

    # ahora construimos respuestas fila a fila
    respuestas = []

    for row_i in range(filas):
        if row_i >= len(filas_groups):
            respuestas.append("")
            continue

        idxs = filas_groups[row_i]
        # en esa fila, asignar cada círculo a la columna más cercana
        scores = {c: 0.0 for c in OPCIONES}

        # elegimos por columna: para cada columna, el círculo más cercano en X
        for ci, letter in enumerate(OPCIONES):
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
                scores[letter] = 0.0
                continue

            x, y, r = best
            scores[letter] = score_circulo(zona_bin, x, y, r)

            # Debug: cajita pequeña en el círculo elegido
            if debug_a4 is not None:
                cx = OMR_REGION["x0"] + x
                cy = OMR_REGION["y0"] + y
                cv2.rectangle(debug_a4, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 0), 2)

        # decisión por scores
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

        # Debug: etiqueta pregunta
        if debug_a4 is not None:
            y_mean = int(np.mean([circles[i][1] for i in idxs]))
            yy = OMR_REGION["y0"] + y_mean
            cv2.putText(
                debug_a4,
                f"{row_i+1}:{resp or '-'}",
                (OMR_REGION["x0"] - 170, yy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if resp == "X" else (0, 0, 0),
                2
            )

    return respuestas, debug_a4


# ============================================================
# PIPELINE PRINCIPAL ✅
# ============================================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) QR ANTES de normalizar (muchas veces se lee mejor)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) normalizar por marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR DESPUÉS de normalizar si no se pudo antes
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

    # 3) binarización
    th = binarizar_tinta_pro(img_a4)

    # 4) filas a leer en esta página
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

    # 5) respuestas por detección REAL de círculos
    respuestas_lista, debug_a4 = detectar_respuestas_por_circulos(img_a4, th, filas, debug=True)
    if not respuestas_lista:
        return {
            "ok": False,
            "error": "No se pudieron detectar burbujas (círculos) en la zona OMR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha": fecha,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr,
            "debug_image": b64jpg(debug_a4, 85) if debug_a4 is not None else None
        }

    # 6) dict global (1..60)
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
