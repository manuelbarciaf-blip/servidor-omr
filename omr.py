from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import re

app = Flask(__name__)

# =========================
# CONFIG PLANTILLA
# =========================
A4_W, A4_H = 2480, 3508                 # A4 a ~300dpi tras normalizar
OPCIONES = ["A", "B", "C", "D"]

# Zona OMR fija dentro del A4 normalizado (ajústala solo si cambias plantilla PDF)
# (estos valores están pensados para tu hoja, con una única columna A-D)
OMR_REGION = {"y0": 650, "y1": 3000, "x0": 780, "x1": 1450}

# La hoja tiene como máximo 30 filas por página (según tu diseño)
MAX_FILAS_POR_HOJA = 30

# Umbrales (robustos para escáner + móvil)
UMBRAL_VACIO = 0.055         # % tinta mínima para considerar marcada (sube si hay ruido)
UMBRAL_DOBLE_RATIO = 0.78    # si 2ª mejor está cerca de la mejor
UMBRAL_DOBLE_ABS = 0.040     # y además supera un mínimo absoluto

# ROI interior (para ignorar borde del cuadrado)
INNER_PAD_X = 0.22
INNER_PAD_Y = 0.22


# =========================
# UTIL: detectar marcas negras y normalizar perspectiva a A4
# =========================
def normalizar_a4_con_marcas(img_bgr):
    """
    Encuentra las 4 marcas negras (esquinas) y aplica perspectiva a A4_W x A4_H.
    Funciona muy bien con escáner y bastante bien con móvil.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # binaria invertida (negro -> blanco)
    _, th = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    # limpieza
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
        # marcas casi cuadradas
        if 0.65 < aspect < 1.35:
            cx = x + bw / 2
            cy = y + bh / 2
            candidates.append((cx, cy, area, x, y, bw, bh))

    if len(candidates) < 4:
        # fallback: al menos escala a A4
        return cv2.resize(img_bgr, (A4_W, A4_H))

    # elegir el mejor candidato para cada esquina por distancia a esa esquina
    corners_targets = {
        "tl": (0, 0),
        "tr": (w, 0),
        "bl": (0, h),
        "br": (w, h),
    }

    chosen = {}
    centers = np.array([[c[0], c[1]] for c in candidates], dtype=np.float32)

    for name, (tx, ty) in corners_targets.items():
        t = np.array([tx, ty], dtype=np.float32)
        d = np.sum((centers - t) ** 2, axis=1)
        idx = int(np.argmin(d))
        chosen[name] = centers[idx]

    tl = chosen["tl"]
    tr = chosen["tr"]
    bl = chosen["bl"]
    br = chosen["br"]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))

    return warped


# =========================
# QR robusto (sin pyzbar)
# =========================
def leer_qr(img_bgr):
    det = cv2.QRCodeDetector()

    # intento 1: imagen completa
    data, _, _ = det.detectAndDecode(img_bgr)
    data = (data or "").strip()
    if data:
        return data

    # intento 2: zona sup izq (donde dices que está el QR)
    crop = img_bgr[0:1200, 0:1200].copy()
    crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    data, _, _ = det.detectAndDecode(crop)
    data = (data or "").strip()
    return data if data else None


def parsear_codigo_qr(codigo):
    """
    Espera algo tipo:
      id_examen|id_alumno|num_preguntas|pagina
    pagina es opcional.
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


# =========================
# Binarización PRO (escáner + móvil)
# =========================
def binarizar_tinta_pro(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # mejorar contraste (muy útil en móvil con sombras)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive threshold (inversa: tinta -> blanco)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # limpieza
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    return th


# =========================
# Cálculo filas según QR + página
# =========================
def filas_a_leer(num_preguntas, pagina):
    if pagina == 1:
        return min(num_preguntas, MAX_FILAS_POR_HOJA), 0
    # página 2
    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0, MAX_FILAS_POR_HOJA
    return min(num_preguntas - MAX_FILAS_POR_HOJA, MAX_FILAS_POR_HOJA), MAX_FILAS_POR_HOJA


# =========================
# Detector OMR por rejilla (1 columna A-D)
# =========================
def detectar_respuestas(zona_bin, filas, debug_color=None):
    """
    zona_bin: binaria (tinta blanca)
    filas: nº filas reales de esta hoja
    debug_color: opcional para dibujar rejilla
    """
    if filas <= 0:
        return [], debug_color

    h, w = zona_bin.shape

    # margen lateral para evitar bordes (ajusta si hace falta)
    margen = int(w * 0.06)
    zona = zona_bin[:, margen:w - margen]
    h2, w2 = zona.shape

    alto_fila = int(h2 / float(MAX_FILAS_POR_HOJA))  # la plantilla usa paso constante (30 filas “base”)
    ancho_op = int(w2 / 4.0)

    respuestas = []

    for i in range(filas):
        # fila i ocupa el "slot" i dentro de los 30 slots base
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

            # ROI interior para ignorar bordes del cuadrado impreso
            ih, iw = celda.shape
            pad_x = int(iw * INNER_PAD_X)
            pad_y = int(ih * INNER_PAD_Y)

            inner = celda[pad_y:ih - pad_y, pad_x:iw - pad_x]
            if inner.size == 0:
                inner = celda

            score = cv2.countNonZero(inner) / float(inner.size)
            dens.append(score)
            rois.append((x0, y0, x1, y1))

        # decisión
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

        # debug overlay
        if debug_color is not None:
            # pintar fila
            for j in range(4):
                x0, yy0, x1, yy1 = rois[j]
                # coordenadas en debug_color (teniendo en cuenta el margen recortado)
                X0 = OMR_REGION["x0"] + margen + x0
                X1 = OMR_REGION["x0"] + margen + x1
                Y0 = OMR_REGION["y0"] + yy0
                Y1 = OMR_REGION["y0"] + yy1

                color = (0, 255, 0)
                cv2.rectangle(debug_color, (X0, Y0), (X1, Y1), color, 2)

                # resaltar elegida
                if resp and resp != "X" and OPCIONES[j] == resp:
                    cv2.rectangle(debug_color, (X0, Y0), (X1, Y1), (0, 0, 255), 3)

            # texto pregunta
            cv2.putText(
                debug_color,
                f"{i+1}:{resp or '-'}",
                (OMR_REGION["x0"] - 160, OMR_REGION["y0"] + (i + 1) * int(alto_fila) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if resp == "X" else (0, 0, 0),
                2
            )

    return respuestas, debug_color


# =========================
# Pipeline principal
# =========================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    # 1) normalizar A4 con marcas
    img_a4 = normalizar_a4_con_marcas(img)

    # 2) leer QR
    codigo = leer_qr(img_a4)
    if not codigo:
        return {"ok": False, "error": "QR no detectado"}

    parsed = parsear_codigo_qr(codigo)
    if not parsed:
        return {"ok": False, "error": "Formato QR inválido", "codigo": codigo}

    id_examen, id_alumno, num_preguntas, pagina = parsed

    # 3) binarizar tinta
    th = binarizar_tinta_pro(img_a4)

    # 4) recortar zona OMR
    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]]
    zona_color = img_a4[OMR_REGION["y0"]:OMR_REGION["y1"], OMR_REGION["x0"]:OMR_REGION["x1"]].copy()

    # debug overlay se pinta sobre una copia del A4 entero (mejor para ver contexto)
    debug_a4 = img_a4.copy()

    # 5) calcular filas de esta hoja
    filas, offset = filas_a_leer(num_preguntas, pagina)
    if filas <= 0:
        return {
            "ok": False,
            "error": "Esta página no tiene preguntas según el QR",
            "codigo": codigo,
            "num_preguntas": num_preguntas,
            "pagina": pagina
        }

    # 6) detectar respuestas
    respuestas_lista, debug_a4 = detectar_respuestas(zona_bin, filas, debug_color=debug_a4)

    # 7) salida como dict por número global (1..60)
    respuestas = {}
    for i, r in enumerate(respuestas_lista, start=1):
        respuestas[str(offset + i)] = r

    # 8) debug images base64
    # debug A4 con rejilla
    _, buff_dbg = cv2.imencode(".jpg", debug_a4, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    debug_image = base64.b64encode(buff_dbg).decode("utf-8")

    # binaria recortada (muy útil)
    _, buff_bin = cv2.imencode(".jpg", zona_bin, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    debug_bin = base64.b64encode(buff_bin).decode("utf-8")

    return {
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "num_preguntas": num_preguntas,
        "pagina": pagina,
        "respuestas": respuestas,
        "debug_image": debug_image,
        "debug_bin": debug_bin
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
