import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# =========================================
# CONFIG
# =========================================
QR_REGION = {
    "x0": 0.00,
    "y0": 0.00,
    "x1": 0.50,
    "y1": 0.40
}

OMR_REGION = {
    "x0": 0.30,
    "y0": 0.22,
    "x1": 0.70,
    "y1": 0.95
}

OPCIONES = ["A", "B", "C", "D"]
PREGUNTAS_POR_HOJA = 20

UMBRAL_VACIO = 0.05
UMBRAL_DOBLE = 0.75

# =========================================
# NORMALIZAR
# =========================================
def normalizar_a4(img):
    return cv2.resize(img, (2480, 3508))

# =========================================
# LECTOR QR ROBUSTO
# =========================================
def leer_qr(img):
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    codes = zbar_decode(gray_full)
    if codes:
        return codes[0].data.decode("utf-8").strip()

    h, w = img.shape[:2]
    x0 = int(w * 0.00)
    y0 = int(h * 0.00)
    x1 = int(w * 0.50)
    y1 = int(h * 0.40)

    qr_crop = img[y0:y1, x0:x1]
    if qr_crop.size == 0:
        return None

    qr_crop = cv2.resize(qr_crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray_qr = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
    gray_qr = cv2.GaussianBlur(gray_qr, (3, 3), 0)

    codes = zbar_decode(gray_qr)
    if not codes:
        return None

    return codes[0].data.decode("utf-8").strip()

# =========================================
# MÁSCARA DE TINTA
# =========================================
def preparar_mascara_tinta(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([70, 10, 10])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_black = np.array([0, 0, 40])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask = cv2.bitwise_or(mask_blue, mask_black)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, th = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    return th

# =========================================
# RECORTE
# =========================================
def recortar(img, region):
    h, w = img.shape[:2]
    return img[
        int(h * region["y0"]):int(h * region["y1"]),
        int(w * region["x0"]):int(w * region["x1"])
    ]

# =========================================
# DETECCIÓN OMR
# =========================================
def detectar_respuestas(zona_bin, zona_color):
    h, w = zona_bin.shape
    alto_fila = int(h / PREGUNTAS_POR_HOJA)
    ancho_op = int(w / 4)

    respuestas = []

    for i in range(PREGUNTAS_POR_HOJA):
        y0 = i * alto_fila
        y1 = y0 + alto_fila

        fila = zona_bin[y0:y1, :]
        densidades = []

        for j in range(4):
            x0 = j * ancho_op
            x1 = (j + 1) * ancho_op
            celda = fila[:, x0:x1]
            densidad = cv2.countNonZero(celda) / celda.size
            densidades.append(densidad)

        max_d = max(densidades)
        idx = densidades.index(max_d)
        segundo = sorted(densidades, reverse=True)[1]

        if max_d < UMBRAL_VACIO:
            respuestas.append("")
        elif segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
        else:
            respuestas.append(OPCIONES[idx])

    return respuestas

# =========================================
# FUNCIÓN PRINCIPAL
# =========================================
def procesar_omr(binario):
    npimg = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "ok": False,
            "error": "Imagen inválida"
        }

    # Normalizar a tamaño A4
    img_a4 = normalizar_a4(img)

    # Leer QR
    codigo = leer_qr(img_a4)
    if not codigo:
        return {
            "ok": False,
            "error": "QR no detectado"
        }

    # Preparar máscara de tinta
    th = preparar_mascara_tinta(img_a4)

    # Recortar zona OMR
    zona_bin = recortar(th, OMR_REGION)
    zona_color = recortar(img_a4, OMR_REGION)

    # Detectar respuestas
    respuestas = detectar_respuestas(zona_bin, zona_color)

    # Generar imagen debug (MUY IMPORTANTE para que la veas en PHP)
    _, buffer = cv2.imencode(".jpg", zona_color)
    debug_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "debug_image": debug_base64
    }
