from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import re

app = Flask(__name__)

# ============================================================
# CONFIG ✅ (A4 normalizado + zonas)
# ============================================================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

# ✅ Zona OMR AMPLIADA POR ARRIBA (y un poco más ancha)
# Ajusta si cambias tu plantilla PDF.
OMR_REGION = {
    "y0": 380,     # ⬅️ más arriba (antes 520/650)
    "y1": 3200,
    "x0": 560,
    "x1": 1750
}

MAX_FILAS_POR_HOJA = 30

# Umbrales lectura tinta (móvil + escáner)
UMBRAL_VACIO = 0.050          # % tinta mínima para considerar marcada
UMBRAL_DOBLE_RATIO = 0.82     # si 2ª opción está muy cerca de la 1ª
UMBRAL_DOBLE_ABS = 0.035      # y además supera un mínimo absoluto

# ROI interior para no contar borde impreso (círculo)
INNER_PAD = 0.28   # 0.25–0.35 suele ir bien


# ============================================================
# UTIL: imagen -> base64 jpg
# ============================================================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


# ============================================================
# 1) NORMALIZAR A4 con marcas negras
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
# 2) QR ROBUSTO (ROI fijo + rotaciones + escalas + preprocess)
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
    out = [img_bgr]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
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

    # 0) Intento en toda la imagen (con variantes)
    for v in _variants(img_bgr):
        s = _try_decode(det, v)
        if s:
            return s, None

    # 1) ROI fijo arriba izquierda (tu QR siempre ahí)
    qr_roi = img_bgr[0:1400, 0:1400].copy()
    debug_qr = b64jpg(qr_roi, 90)

    scales = [1.0, 1.6, 2.2, 2.8, 3.4]
    rotations = [
        lambda im: im,
        lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
        lambda im: cv2.rotate(im, cv2.ROTATE_180),
        lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    for sc in scales:
        roi = qr_roi
        if sc != 1.0:
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        for rot in rotations:
            rr = rot(roi)
            for v in _variants(rr):
                s = _try_decode(det, v)
                if s:
                    return s, debug_qr

    # 2) ROI grande por si el QR está un poco más metido
    big = img_bgr[0:1700, 0:1700].copy()
    for v in _variants(big):
        s = _try_decode(det, v)
        if s:
            return s, debug_qr

    return None, debug_qr


def parsear_codigo_qr(codigo):
    """
    ✅ Tu formato real:
      id_examen|id_alumno|fecha|num_preguntas
    opcional:
      |pagina

    Soporta también:
      id_examen|id_alumno|num_preguntas|pagina   (legacy)
    y fallback por regex si viene texto mezclado.
    """
    if not codigo:
        return None

    partes = [p.strip() for p in codigo.split("|") if p.strip()]

    # Caso normal con |
    if len(partes) >= 4:
        # Si 3ª parte parece fecha (tiene '-'), entonces:
        # 1=id_examen, 2=id_alumno, 3=fecha, 4=num_preguntas, 5=pagina?
        if "-" in partes[2] and partes[3].isdigit():
            try:
                id_examen = int(partes[0])
                id_alumno = int(partes[1])
                fecha = partes[2]
                num_preg = int(partes[3])
                pagina = int(partes[4]) if len(partes) >= 5 and partes[4].isdigit() else 1
                pagina = 1 if pagina not in (1, 2) else pagina
                return id_examen, id_alumno, fecha, num_preg, pagina
            except:
                pass

        # Legacy: 3ª num_preg
        if partes[2].isdigit():
            try:
                id_examen = int(partes[0])
                id_alumno = int(partes[1])
                fecha = None
                num_preg = int(partes[2])
                pagina = int(partes[3]) if len(partes) >= 4 and partes[3].isdigit() else 1
                pagina = 1 if pagina not in (1, 2) else pagina
                return id_examen, id_alumno, fecha, num_preg, pagina
            except:
                pass

    # Fallback regex (si el QR trae texto con números)
    nums = re.findall(r"\d+", codigo)
    # mínimo: examen, alumno, (año), (mes), (día), num_preg...
    # aquí no adivinamos fecha segura; al menos sacamos examen/alumno/num_preg si se puede.
    if len(nums) >= 3:
        try:
            id_examen = int(nums[0])
            id_alumno = int(nums[1])
            num_preg = int(nums[-1])  # normalmente el último
            fecha = None
            pagina = 1
            return id_examen, id_alumno, fecha, num_preg, pagina
        except:
            pass

    return None


# ============================================================
# 3) BINARIZACIÓN tinta (móvil/escáner)
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
# 4) filas según QR + página (hasta 60 => 2 hojas)
# ============================================================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# ============================================================
# 5) DETECTAR CÍRCULOS + CLUSTER columnas/filas (sin rejilla fija)
# ============================================================
def _kmeans_1d(values, k):
    """k-means 1D simple usando cv2.kmeans"""
    vals = np.array(values, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.flatten()
    return labels.flatten(), centers


def detectar_circulos(zona_gray):
    """
    Devuelve lista de (x,y,r) en coordenadas de la zona recortada.
    """
    g = cv2.GaussianBlur(zona_gray, (7, 7), 0)

    # HoughCircles suele funcionar muy bien con estos círculos
    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=28,
        param1=120,
        param2=28,
        minRadius=10,
        maxRadius=28
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")
    # filtrar duplicados cercanos
    out = []
    for (x, y, r) in circles:
        if r <= 0:
            continue
        ok = True
        for (xx, yy, rr) in out:
            if (x - xx) ** 2 + (y - yy) ** 2 < 16**2:
                ok = False
                break
        if ok:
            out.append((x, y, r))
    return out


def detectar_respuestas_por_circulos(img_a4, th_a4, filas, debug_a4=None):
    """
    - Recorta zona OMR
    - Detecta círculos reales
    - Cluster en 4 columnas por X
    - Cluster en 'filas' por Y
    - Evalúa tinta en ROI interior del círculo
    """
    if filas <= 0:
        return [], debug_a4

    x0, x1 = OMR_REGION["x0"], OMR_REGION["x1"]
    y0, y1 = OMR_REGION["y0"], OMR_REGION["y1"]

    zona_gray = cv2.cvtColor(img_a4[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    zona_th   = th_a4[y0:y1, x0:x1]

    circles = detectar_circulos(zona_gray)
    if len(circles) < max(8, filas * 2):
        # no hay suficientes círculos detectados
        return None, debug_a4

    xs = [c[0] for c in circles]
    ys = [c[1] for c in circles]

    # 4 columnas por X
    labels_x, centers_x = _kmeans_1d(xs, 4)
    order_cols = np.argsort(centers_x)  # izquierda->derecha
    # mapa label->col_index (0..3)
    label_to_col = {int(order_cols[i]): i for i in range(4)}

    # filas por Y (según QR)
    labels_y, centers_y = _kmeans_1d(ys, filas)
    order_rows = np.argsort(centers_y)  # arriba->abajo
    label_to_row = {int(order_rows[i]): i for i in range(filas)}

    # agrupar círculos por (row, col): elegir el más cercano al centro de ese cluster
    grid = {}
    for idx, (x, y, r) in enumerate(circles):
        lx = int(labels_x[idx])
        ly = int(labels_y[idx])
        col = label_to_col.get(lx, None)
        row = label_to_row.get(ly, None)
        if col is None or row is None:
            continue

        # distancia al centro del cluster (para quedarnos con el mejor)
        dx = x - centers_x[lx]
        dy = y - centers_y[ly]
        d2 = float(dx*dx + dy*dy)

        key = (row, col)
        if key not in grid or d2 < grid[key]["d2"]:
            grid[key] = {"x": x, "y": y, "r": r, "d2": d2}

    respuestas = []
    for row in range(filas):
        scores = []
        coords = []
        for col in range(4):
            cell = grid.get((row, col))
            if not cell:
                scores.append(0.0)
                coords.append(None)
                continue

            cx, cy, r = int(cell["x"]), int(cell["y"]), int(cell["r"])
            # ROI interior circular aproximada con cuadrado interior
            pad = int(r * INNER_PAD)
            rx0 = max(cx - r + pad, 0)
            rx1 = min(cx + r - pad, zona_th.shape[1] - 1)
            ry0 = max(cy - r + pad, 0)
            ry1 = min(cy + r - pad, zona_th.shape[0] - 1)

            roi = zona_th[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                score = 0.0
            else:
                score = cv2.countNonZero(roi) / float(roi.size)

            scores.append(float(score))
            coords.append((cx, cy, r))

        max_d = max(scores)
        best = int(np.argmax(scores))
        second = sorted(scores, reverse=True)[1] if len(scores) >= 2 else 0.0

        if max_d < UMBRAL_VACIO:
            resp = ""
        elif (second > UMBRAL_DOBLE_ABS) and (second > max_d * UMBRAL_DOBLE_RATIO):
            resp = "X"
        else:
            resp = OPCIONES[best]

        respuestas.append(resp)

        if debug_a4 is not None:
            # dibujar círculos
            for col in range(4):
                c = coords[col]
                if not c:
                    continue
                cx, cy, r = c
                cv2.circle(debug_a4, (x0 + cx, y0 + cy), r, (0, 255, 0), 2)
            # resaltar la elegida
            if resp and resp != "X":
                c = coords[best]
                if c:
                    cx, cy, r = c
                    cv2.circle(debug_a4, (x0 + cx, y0 + cy), r, (0, 0, 255), 3)

            # etiqueta pregunta
            cv2.putText(
                debug_a4,
                f"{row+1}:{resp or '-'}",
                (x0 - 210, y0 + int(centers_y[order_rows[row]]) + 8),
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

    # 1) Normalizar A4 por marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR después de normalizar si falló
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

    id_examen, id_alumno, fecha_examen, num_preguntas, pagina = parsed

    # 3) binarizar tinta
    th = binarizar_tinta_pro(img_a4)

    # 4) filas según QR
    filas, offset = filas_a_leer(num_preguntas, pagina)
    if filas <= 0:
        return {
            "ok": False,
            "error": "Esta página no tiene preguntas según el QR",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha_examen": fecha_examen,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr
        }

    debug_a4 = img_a4.copy()

    # dibujar zona OMR para ver encuadre
    cv2.rectangle(
        debug_a4,
        (OMR_REGION["x0"], OMR_REGION["y0"]),
        (OMR_REGION["x1"], OMR_REGION["y1"]),
        (255, 0, 0),
        3
    )

    # 5) detectar respuestas por círculos (sin rejilla fija)
    respuestas_lista, debug_a4 = detectar_respuestas_por_circulos(img_a4, th, filas, debug_a4)

    if respuestas_lista is None:
        # fallback: si Hough fallara algún día, devolvemos error + debug
        return {
            "ok": False,
            "error": "No se pudieron detectar los círculos OMR (HoughCircles)",
            "codigo": codigo,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha_examen": fecha_examen,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_image": b64jpg(debug_a4, 85),
            "debug_qr": debug_qr
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
        "fecha_examen": fecha_examen,
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
