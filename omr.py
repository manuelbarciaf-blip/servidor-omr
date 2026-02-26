import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# =========================================
# CONFIG (OPTIMIZADO PARA TU PLANTILLA FIJA)
# =========================================
PREGUNTAS_POR_HOJA = 20
OPCIONES = ["A", "B", "C", "D"]

# Zona real tras normalizar a A4 (calibrada para tu hoja)
OMR_REGION_PIX = {
    "y0": 650,
    "y1": 3000,
    "x0": 780,
    "x1": 1450
}

# Umbrales ajustados para bolígrafo azul/negro en fotos móviles
UMBRAL_VACIO = 0.025
UMBRAL_DOBLE = 0.70

# =========================================
# NORMALIZAR A4 CON CORRECCIÓN DE PERSPECTIVA
# (CLAVE para fotos con móvil)
# =========================================
def normalizar_a4(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar zonas negras (esquinas de tu plantilla)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)[1]
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 4000:  # esquinas negras grandes
            x, y, w, h = cv2.boundingRect(c)
            candidatos.append((x, y, w, h))

    # Si no detecta esquinas (fallback seguro)
    if len(candidatos) < 4:
        return cv2.resize(img, (2480, 3508))

    # Centros de las esquinas
    puntos = []
    for (x, y, w, h) in candidatos:
        puntos.append([x + w // 2, y + h // 2])

    pts = np.array(puntos, dtype="float32")

    # Ordenar esquinas correctamente
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    dst = np.array([
        [0, 0],
        [2480, 0],
        [2480, 3508],
        [0, 3508]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (2480, 3508))

    return warped


# =========================================
# LECTOR QR ROBUSTO (optimizado para tu hoja)
# =========================================
def leer_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Intento 1: imagen completa
    codes = zbar_decode(gray)
    if codes:
        return codes[0].data.decode("utf-8").strip()

    # Intento 2: zona superior izquierda (donde está tu QR)
    qr_crop = img[0:1200, 0:1200]

    qr_crop = cv2.resize(qr_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray_qr = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
    gray_qr = cv2.GaussianBlur(gray_qr, (3, 3), 0)

    codes = zbar_decode(gray_qr)
    if not codes:
        return None

    return codes[0].data.decode("utf-8").strip()


# =========================================
# MÁSCARA DE TINTA (AZUL + NEGRO)
# =========================================
def preparar_mascara_tinta(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Azul (bolígrafo típico)
    lower_blue = np.array([80, 30, 30])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Negro / gris oscuro
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask = cv2.bitwise_or(mask_blue, mask_black)

    # Limpieza de ruido
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, th = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    return th


# =========================================
# RECORTE OMR (COORDENADAS FIJAS = SIN DESFASE)
# =========================================
def recortar_omr(img):
    return img[
        OMR_REGION_PIX["y0"]:OMR_REGION_PIX["y1"],
        OMR_REGION_PIX["x0"]:OMR_REGION_PIX["x1"]
    ]


# =========================================
# DETECCIÓN DE RESPUESTAS (PRECISA Y ESTABLE)
# =========================================
def detectar_respuestas(zona_bin):
    h, w = zona_bin.shape

    # Eliminar márgenes laterales (evita desplazamientos A→B)
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
        idx = densidades.index(max_d)
        segundo = sorted(densidades, reverse=True)[1]

        # Lógica robusta OMR
        if max_d < UMBRAL_VACIO:
            respuestas.append("")          # en blanco
        elif segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")         # doble marca
        else:
            respuestas.append(OPCIONES[idx])

    return respuestas


# =========================================
# FUNCIÓN PRINCIPAL (LA QUE USA PHP)
# =========================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "ok": False,
            "error": "Imagen inválida"
        }

    # 1️⃣ Normalizar hoja (perspectiva + A4 real)
    img_a4 = normalizar_a4(img)

    # 2️⃣ Leer QR
    codigo = leer_qr(img_a4)
    if not codigo:
        return {
            "ok": False,
            "error": "QR no detectado"
        }

    # 3️⃣ Detectar tinta (azul/negro)
    mascara = preparar_mascara_tinta(img_a4)

    # 4️⃣ Recortar zona OMR estable
    zona_bin = recortar_omr(mascara)
    zona_color = recortar_omr(img_a4)

    # 5️⃣ Detectar respuestas
    respuestas = detectar_respuestas(zona_bin)

    # 6️⃣ Imagen debug (para tu visor PHP)
    debug = cv2.cvtColor(zona_color, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    debug_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "debug_image": debug_base64
    }
