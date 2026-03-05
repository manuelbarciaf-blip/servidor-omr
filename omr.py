from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import re

app = Flask(__name__)

# ============================================================
# CONFIG ✅ (ajustada a tu hoja real)
# ============================================================
A4_W, A4_H = 2480, 3508
OPCIONES = ["A", "B", "C", "D"]

# ✅ Ampliada por arriba (tu comentario)
# Ajuste recomendado según tu última captura: subir y0 y ampliar y1 un poco
OMR_REGION = {
    "y0": 430,     # antes 520
    "y1": 3250,    # antes 3150
    "x0": 600,     # antes 620
    "x1": 1700     # antes 1680
}

# La plantilla usa slots base de 30 filas por hoja
MAX_FILAS_POR_HOJA = 30

# Umbrales para tinta (móvil/escáner)
UMBRAL_VACIO = 0.050
UMBRAL_DOBLE_RATIO = 0.80
UMBRAL_DOBLE_ABS = 0.035

# ROI interior dentro de cada celda (para no contar el borde)
INNER_PAD_X = 0.22
INNER_PAD_Y = 0.22

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

    # negro -> blanco (invertida)
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
# 2) QR ROBUSTO (multi-intentos + ROI fijo + warp si hay puntos)
# ============================================================
def _try_decode(det, img_bgr):
    # Multi si existe
    try:
        ok, decoded_info, points, _ = det.detectAndDecodeMulti(img_bgr)
        if ok and decoded_info:
            for s in decoded_info:
                s = (s or "").strip()
                if s:
                    return s
    except Exception:
        pass

    s, pts, _ = det.detectAndDecode(img_bgr)
    s = (s or "").strip()
    if s:
        return s
    return None

def _variants(img_bgr):
    out = [img_bgr]
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

def _warp_qr_if_points(det, img_bgr):
    """
    Si detecta QR (puntos) pero no decodifica, intentamos enderezarlo (warp) y re-decodificar.
    """
    ok = det.detect(img_bgr)[0] if isinstance(det.detect(img_bgr), tuple) else False
    try:
        ok, pts = det.detect(img_bgr)
    except Exception:
        return None
    if not ok or pts is None:
        return None
    pts = pts.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] != 4:
        return None

    # ordenar puntos
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    size = 600
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (size, size))

    # probar varias variantes del warp
    for v in _variants(warped):
        s = _try_decode(det, v)
        if s:
            return s
    return None

def leer_qr_robusto(img_bgr):
    det = cv2.QRCodeDetector()

    # 0) Imagen completa (y warp si detecta puntos)
    for v in _variants(img_bgr):
        s = _try_decode(det, v)
        if s:
            return s, None
        sw = _warp_qr_if_points(det, v)
        if sw:
            return sw, None

    # 1) ROI fijo sup-izq (tu QR SIEMPRE ahí)
    qr_roi = img_bgr[0:1300, 0:1300].copy()
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
                sw = _warp_qr_if_points(det, v)
                if sw:
                    return sw, debug_qr

    # 2) ROI grande con más contexto
    big = img_bgr[0:1700, 0:1700].copy()
    for v in _variants(big):
        s = _try_decode(det, v)
        if s:
            return s, debug_qr
        sw = _warp_qr_if_points(det, v)
        if sw:
            return sw, debug_qr

    return None, debug_qr

def parsear_codigo_qr(codigo):
    """
    Formato real tuyo:
      id_examen|id_alumno|fecha|num_preguntas
    (pagina opcional al final si algún día la añades)
    """
    if not codigo:
        return None

    if "|" in codigo:
        partes = [p.strip() for p in codigo.split("|") if p.strip()]
        if len(partes) >= 4:
            try:
                id_examen = int(partes[0])
                id_alumno = int(partes[1])
                fecha = partes[2]  # string yyyy-mm-dd
                num_preg = int(partes[3])
                pagina = int(partes[4]) if len(partes) >= 5 and partes[4].isdigit() else 1
                pagina = 1 if pagina not in (1, 2) else pagina
                return id_examen, id_alumno, fecha, num_preg, pagina
            except:
                pass

    # fallback: extraer enteros (por si el QR trae texto extra)
    nums = re.findall(r"\d+", codigo)
    # esperamos al menos id_examen, id_alumno, yyyy, mm, dd, num_preg
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
# 5) AUTO-COLUMNAS (para no confundir A/B/C/D)
# ============================================================
def detectar_columnas_auto(zona_gray):
    """
    Encuentra 4 columnas usando proyección vertical de bordes.
    Devuelve una lista de 5 límites [x0, x1, x2, x3, x4] (4 celdas).
    """
    # Bordes -> las circunferencias destacan mucho
    edges = cv2.Canny(zona_gray, 40, 120)
    # suavizar perfil
    prof = edges.sum(axis=0).astype(np.float32)

    # normalizar y suavizar
    prof = prof / (prof.max() + 1e-6)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 61), 0).reshape(-1)

    w = zona_gray.shape[1]
    # buscamos 4 picos en la zona izquierda (donde están las burbujas) pero sin asumir demasiado:
    # tomamos el 70% izquierdo como región probable
    left_w = int(w * 0.72)
    prof2 = prof[:left_w]

    # escoger candidatos: puntos con valor alto
    idxs = np.argsort(prof2)[::-1]  # desc
    peaks = []
    min_sep = int(w * 0.08)  # separación mínima entre picos

    for i in idxs:
        if prof2[i] < 0.25:
            break
        if all(abs(i - p) > min_sep for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= 4:
            break

    # fallback: dividir en 4 si no detecta bien
    if len(peaks) < 4:
        return [0, w//4, w//2, 3*w//4, w]

    peaks = sorted(peaks)
    # convertir centros a límites (midpoints)
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(int((a + b) / 2))
    bounds.append(w)

    # ahora bounds tiene 5 elementos idealmente, si no, fallback
    if len(bounds) != 5:
        return [0, w//4, w//2, 3*w//4, w]
    return bounds

# ============================================================
# 6) DETECTOR OMR (rejilla estable 1 columna A-D)
# ============================================================
def detectar_respuestas(zona_bin, zona_gray, filas, debug_a4=None):
    if filas <= 0:
        return [], debug_a4

    h, w = zona_bin.shape

    # margen lateral para evitar marco
    margen = int(w * 0.04)
    zb = zona_bin[:, margen:w - margen]
    zg = zona_gray[:, margen:w - margen]
    h2, w2 = zb.shape

    alto_slot = int(h2 / float(MAX_FILAS_POR_HOJA))

    # ✅ auto columnas
    col_bounds = detectar_columnas_auto(zg)

    respuestas = []
    for i in range(filas):
        y0 = i * alto_slot
        y1 = (i + 1) * alto_slot
        fila = zb[y0:y1, :]

        dens = []
        rois = []

        for j in range(4):
            x0 = col_bounds[j]
            x1 = col_bounds[j + 1]
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

        # debug overlay
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
                (OMR_REGION["x0"] - 200, OMR_REGION["y0"] + (i + 1) * alto_slot - 10),
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

    # 0) QR antes de normalizar (a veces mejor)
    codigo0, debug_qr0 = leer_qr_robusto(img)
    parsed0 = parsear_codigo_qr(codigo0) if codigo0 else None

    # 1) normalizar A4 con marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) QR después de normalizar si falló antes
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

    # 4) recorte OMR
    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]
    zona_gray = cv2.cvtColor(img_a4[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]], cv2.COLOR_BGR2GRAY)

    # debug A4
    debug_a4 = img_a4.copy()
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
            "fecha_examen": fecha_examen,
            "num_preguntas": num_preguntas,
            "pagina": pagina,
            "debug_qr": debug_qr
        }

    # 6) detectar respuestas (con auto columnas)
    respuestas_lista, debug_a4 = detectar_respuestas(zona_bin, zona_gray, filas, debug_a4)

    # 7) salida dict global
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
