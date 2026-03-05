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

# ✅ Zona OMR (la ampliada que te estaba funcionando mejor)
OMR_REGION = {
    "y0": 520,
    "y1": 3150,
    "x0": 620,
    "x1": 1680
}

MAX_FILAS_POR_HOJA = 30

# Umbrales lectura (ajústalos si quieres)
UMBRAL_VACIO = 0.050
UMBRAL_DOBLE_RATIO = 0.80
UMBRAL_DOBLE_ABS = 0.035

# Para ignorar el borde impreso del círculo
CIRCLE_INNER_SHRINK = 0.62  # 0.55–0.70 suele ir bien


# ============================================================
# UTIL: imagen -> base64 jpg
# ============================================================
def b64jpg(img_bgr, quality=85):
    ok, buff = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buff).decode("utf-8")


# ============================================================
# 1) NORMALIZAR A4 con marcas negras (robusto)
# ============================================================
def normalizar_a4_con_marcas(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binaria invertida: negro => blanco
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

    # Fallback: escala a A4
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
# 2) QR ROBUSTO (multi intentos + ROI fijo + rotaciones)
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

    for v in _variants(img_bgr):
        s = _try_decode(det, v)
        if s:
            return s, None

    # ROI fijo arriba izquierda (tu QR está SIEMPRE ahí)
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
# 5A) Detectar círculos (centros reales) dentro de OMR_REGION ✅
# ============================================================
def detectar_circulos_en_omr(omr_bgr):
    """
    Devuelve lista de círculos [(cx, cy, r), ...] en coordenadas del recorte OMR.
    Basado en contornos con circularidad (mejor que dividir en 4).
    """
    gray = cv2.cvtColor(omr_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Umbral para capturar trazos oscuros (círculos impresos)
    # (invertimos para que el círculo sea blanco)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    h, w = gray.shape[:2]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:  # descarta ruido
            continue

        per = cv2.arcLength(c, True)
        if per <= 0:
            continue

        circularity = (4.0 * np.pi * area) / (per * per + 1e-6)
        if circularity < 0.55:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r < 10 or r > 60:  # rango de radios razonable en A4 normalizado
            continue

        cx, cy = float(x), float(y)
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue

        circles.append((cx, cy, r))

    # Filtra un poco: nos quedamos con los más consistentes por radio (evita falsos)
    if len(circles) < 10:
        return circles

    rs = np.array([c[2] for c in circles], dtype=np.float32)
    med = float(np.median(rs))
    lo = med * 0.65
    hi = med * 1.45
    circles = [c for c in circles if lo <= c[2] <= hi]

    return circles


def kmeans_1d(values, k=4):
    vals = np.array(values, dtype=np.float32).reshape(-1, 1)
    if len(vals) < k:
        return None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, flags)
    centers = centers.flatten()
    order = np.argsort(centers)
    centers_sorted = centers[order]
    # remapeo labels a orden A..D
    label_map = {int(order[i]): i for i in range(k)}
    labels_sorted = np.array([label_map[int(l)] for l in labels.flatten()], dtype=np.int32)
    return centers_sorted, labels_sorted


def agrupar_por_filas(circles, y_eps):
    """
    Agrupa círculos por filas usando tolerancia en Y.
    circles: [(cx, cy, r)...] (coords OMR crop)
    """
    circles_sorted = sorted(circles, key=lambda c: c[1])
    rows = []
    for c in circles_sorted:
        if not rows:
            rows.append([c])
            continue
        if abs(c[1] - np.mean([x[1] for x in rows[-1]])) <= y_eps:
            rows[-1].append(c)
        else:
            rows.append([c])
    return rows


# ============================================================
# 5B) Leer respuestas usando círculos detectados ✅
# ============================================================
def leer_respuestas_por_circulos(omr_bin, circles, filas, debug_a4=None):
    """
    omr_bin: binaria (tinta blanca) SOLO del recorte OMR
    circles: lista de círculos detectados en recorte OMR (cx,cy,r)
    filas: nº preguntas a leer en esta hoja (<=30)
    """
    if filas <= 0:
        return [], debug_a4, False

    if len(circles) < filas * 3:
        # muy pocos círculos -> mala detección
        return [], debug_a4, False

    # 1) columnas reales (kmeans sobre X)
    xs = [c[0] for c in circles]
    col_centers, col_labels = kmeans_1d(xs, k=4)
    if col_centers is None:
        return [], debug_a4, False

    # 2) agrupar por filas (eps basado en radio mediano)
    r_med = float(np.median([c[2] for c in circles]))
    y_eps = max(18.0, r_med * 1.7)
    rows = agrupar_por_filas(circles, y_eps=y_eps)

    # Nos quedamos con filas que tengan suficientes círculos
    # (deberían tener ~4)
    rows = [row for row in rows if len(row) >= 3]
    rows = sorted(rows, key=lambda row: np.mean([c[1] for c in row]))

    if len(rows) < filas:
        return [], debug_a4, False

    # Limitamos al número de filas reales a leer (desde arriba)
    rows = rows[:filas]

    respuestas = []
    h, w = omr_bin.shape[:2]

    for i, row in enumerate(rows):
        # Para cada columna, elegimos el círculo más cercano al centro de esa columna
        row_sorted = []
        for j in range(4):
            target_x = col_centers[j]
            best = min(row, key=lambda c: abs(c[0] - target_x))
            row_sorted.append(best)

        dens = []

        for j, (cx, cy, r) in enumerate(row_sorted):
            # máscara circular interior
            rr = max(6, int(r * CIRCLE_INNER_SHRINK))
            cx_i = int(round(cx))
            cy_i = int(round(cy))

            x0 = max(0, cx_i - rr)
            x1 = min(w - 1, cx_i + rr)
            y0 = max(0, cy_i - rr)
            y1 = min(h - 1, cy_i + rr)

            roi = omr_bin[y0:y1, x0:x1]
            if roi.size == 0:
                dens.append(0.0)
                continue

            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx_i - x0, cy_i - y0), rr, 255, -1)

            ink = cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))
            area = cv2.countNonZero(mask)
            score = float(ink) / float(area + 1e-6)
            dens.append(score)

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

        # Debug overlay: dibuja círculos detectados + elegido
        if debug_a4 is not None:
            for j, (cx, cy, r) in enumerate(row_sorted):
                X = int(OMR_REGION["x0"] + cx)
                Y = int(OMR_REGION["y0"] + cy)
                R = int(r)
                cv2.circle(debug_a4, (X, Y), R, (0, 255, 0), 2)
                if resp and resp != "X" and OPCIONES[j] == resp:
                    cv2.circle(debug_a4, (X, Y), R, (0, 0, 255), 3)

            cv2.putText(
                debug_a4,
                f"{i+1}:{resp or '-'}",
                (OMR_REGION["x0"] - 210, int(OMR_REGION["y0"] + np.mean([c[1] for c in row]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if resp == "X" else (0, 0, 0),
                2
            )

    return respuestas, debug_a4, True


# ============================================================
# 5C) Fallback: tu método anterior por “rejilla” (por si falla)
# ============================================================
def detectar_respuestas_fallback(zona_bin, filas, debug_a4=None):
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape
    margen = int(w * 0.05)
    zona = zona_bin[:, margen:w - margen]
    h2, w2 = zona.shape

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
            pad_x = int(iw * 0.23)
            pad_y = int(ih * 0.23)
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

                cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 255, 0), 1)
                if resp and resp != "X" and OPCIONES[j] == resp:
                    cv2.rectangle(debug_a4, (X0, Y0), (X1, Y1), (0, 0, 255), 2)

    return respuestas, debug_a4


# ============================================================
# PIPELINE PRINCIPAL ✅ (QR imprescindible)
# ============================================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 0) Intento QR ANTES de normalizar
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) Normalizar por marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR DESPUÉS de normalizar si hace falta
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

    # 3) Binarización tinta
    th = binarizar_tinta_pro(img_a4)

    # 4) Recorte zona OMR (binaria + color)
    omr_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]
    omr_bgr = img_a4[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]].copy()

    debug_a4 = img_a4.copy()

    # Dibuja OMR_REGION
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

    # 6) ✅ Lectura por círculos detectados (corrige el “desfase a B”)
    circles = detectar_circulos_en_omr(omr_bgr)
    respuestas_lista, debug_a4, ok_circles = leer_respuestas_por_circulos(
        omr_bin, circles, filas, debug_a4=debug_a4
    )

    # 7) Fallback si falló
    if not ok_circles:
        respuestas_lista, debug_a4 = detectar_respuestas_fallback(omr_bin, filas, debug_a4=debug_a4)

    # 8) dict global (1..60)
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
        "debug_qr": debug_qr,
        "debug_circles": len(circles),
        "modo": "circulos" if ok_circles else "fallback"
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
