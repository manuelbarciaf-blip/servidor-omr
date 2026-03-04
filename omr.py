import cv2
import numpy as np
import base64

PREGUNTAS_POR_HOJA = 20
OPCIONES = ["A", "B", "C", "D"]

OMR_REGION_PIX = {"y0": 650, "y1": 3000, "x0": 780, "x1": 1450}

UMBRAL_VACIO = 0.025
UMBRAL_DOBLE_RATIO = 0.70
UMBRAL_DOBLE_ABS = 0.018   # ✅ nuevo: mínimo absoluto para "doble"

A4_W, A4_H = 2480, 3508


def _corner_like_boxes(thresh):
    # contornos grandes (esquinas)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h else 0
        if 0.65 < aspect < 1.35:  # bastante cuadrado
            boxes.append((x, y, w, h, area))
    return boxes


def normalizar_a4(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    boxes = _corner_like_boxes(th)

    if len(boxes) < 4:
        # fallback seguro
        return cv2.resize(img_bgr, (A4_W, A4_H))

    h, w = gray.shape[:2]
    corners_target = np.array([
        [0, 0],        # TL
        [w, 0],        # TR
        [0, h],        # BL
        [w, h],        # BR
    ], dtype="float32")

    centers = np.array([[x + bw / 2, y + bh / 2] for x, y, bw, bh, _ in boxes], dtype="float32")

    # ✅ Elegir el mejor candidato para cada esquina por distancia
    chosen = []
    for c in corners_target:
        d = np.sum((centers - c) ** 2, axis=1)
        chosen.append(centers[np.argmin(d)])

    rect = np.array(chosen, dtype="float32")

    # ordenar TL,TR,BL,BR
    s = rect.sum(axis=1)
    tl = rect[np.argmin(s)]
    br = rect[np.argmax(s)]
    diff = np.diff(rect, axis=1)
    tr = rect[np.argmin(diff)]
    bl = rect[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype="float32")
    dst = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (A4_W, A4_H))

    return warped


def leer_qr_opencv(img_bgr):
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img_bgr)
    data = (data or "").strip()
    return data if data else None


def preparar_mascara_tinta(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # azul bolígrafo
    lower_blue = np.array([80, 30, 30])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # negro/ gris oscuro (útil en escaneo)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # extra: umbral por gris (cuando el escáner es casi B/N)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.bitwise_or(mask_blue, mask_black)
    mask = cv2.bitwise_or(mask, mask_gray)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    _, th = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    return th


def recortar_omr(img):
    return img[OMR_REGION_PIX["y0"]:OMR_REGION_PIX["y1"], OMR_REGION_PIX["x0"]:OMR_REGION_PIX["x1"]]


def detectar_respuestas(zona_bin):
    h, w = zona_bin.shape

    margen = int(w * 0.08)
    zona_util = zona_bin[:, margen:w - margen]

    h2, w2 = zona_util.shape
    alto_fila = int(h2 / PREGUNTAS_POR_HOJA)
    ancho_op = int(w2 / 4)

    respuestas = []

    for i in range(PREGUNTAS_POR_HOJA):
        y0 = i * alto_fila
        y1 = y0 + alto_fila
        fila = zona_util[y0:y1, :]

        densidades = []
        for j in range(4):
            x0 = j * ancho_op
            x1 = (j + 1) * ancho_op
            celda = fila[:, x0:x1]

            if celda.size == 0:
                densidades.append(0)
                continue

            densidad = cv2.countNonZero(celda) / float(celda.size)
            densidades.append(densidad)

        max_d = max(densidades)
        idx = int(np.argmax(densidades))
        segundo = sorted(densidades, reverse=True)[1]

        if max_d < UMBRAL_VACIO:
            respuestas.append("")
        elif (segundo > UMBRAL_DOBLE_ABS) and (segundo > max_d * UMBRAL_DOBLE_RATIO):
            respuestas.append("X")
        else:
            respuestas.append(OPCIONES[idx])

    return respuestas


def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    img_a4 = normalizar_a4(img)

    codigo = leer_qr_opencv(img_a4)
    if not codigo:
        return {"ok": False, "error": "QR no detectado"}

    mascara = preparar_mascara_tinta(img_a4)

    zona_bin = recortar_omr(mascara)
    zona_color = recortar_omr(img_a4)

    respuestas = detectar_respuestas(zona_bin)

    # debug: zona_color + zona_bin
    debug_color = cv2.cvtColor(zona_color, cv2.COLOR_BGR2RGB)
    _, buff1 = cv2.imencode(".jpg", debug_color, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    debug_image = base64.b64encode(buff1).decode("utf-8")

    _, buff2 = cv2.imencode(".jpg", zona_bin, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    debug_bin = base64.b64encode(buff2).decode("utf-8")

    return {
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "debug_image": debug_image,
        "debug_bin": debug_bin
    }
