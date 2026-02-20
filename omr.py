import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# VALORES OMR DEFINIDOS POR MANUEL
# ---------------------------------------------------------
VALORES_OMR = {
    "x0": 0.25,
    "y0": 0.26,
    "x1": 0.40,
    "y1": 0.77
}

# ---------------------------------------------------------
# LECTURA QR EN TODA LA IMAGEN (RECOMENDADO)
# ---------------------------------------------------------
def leer_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Buscar QR en toda la imagen
    codes = zbar_decode(blur)

    if not codes:
        return None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# RECORTE PORCENTUAL DE BURBUJAS
# ---------------------------------------------------------
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    X0 = int(w * valores["x0"])
    Y0 = int(h * valores["y0"])
    X1 = int(w * valores["x1"])
    Y1 = int(h * valores["y1"])
    return img[Y0:Y1, X0:X1]

# ---------------------------------------------------------
# DETECCIÓN DE 20 PREGUNTAS
# ---------------------------------------------------------
def detectar_respuestas_20(zona):
    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    filas = 20
    opciones = 4

    h, w = th.shape
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    for fila in range(filas):
        y0 = fila * alto_fila
        y1 = (fila + 1) * alto_fila
        fila_img = th[y0:y1, :]

        valores = []
        for o in range(opciones):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op
            celda = fila_img[:, x0:x1]
            negro = cv2.countNonZero(celda)
            valores.append(negro)

        max_val = max(valores)
        media = np.mean(valores)

        if max_val < media * 1.5:
            respuestas.append(None)
        else:
            respuestas.append(letras[valores.index(max_val)])

    return respuestas

# ---------------------------------------------------------
# PROCESAR IMAGEN COMPLETA
# ---------------------------------------------------------
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "No se pudo decodificar la imagen"}

    # --- LECTURA QR EN TODA LA IMAGEN ---
    id_examen, id_alumno, fecha_qr = leer_qr(img)

    # --- DEBUG QR (imagen completa) ---
    _, qr_buf = cv2.imencode(".jpg", img)
    qr_debug = base64.b64encode(qr_buf).decode()

    # --- Recorte de burbujas ---
    zona = recortar_porcentual(img, VALORES_OMR)
    respuestas = detectar_respuestas_20(zona)

    # --- Debug de burbujas ---
    _, buffer = cv2.imencode(".jpg", zona)
    debug_b64 = base64.b64encode(buffer).decode()

    return {
        "ok": True,
        "codigo": f"{id_examen}|{id_alumno}|{fecha_qr}" if id_examen else None,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha_qr": fecha_qr,
        "respuestas": respuestas,
        "debug_image": debug_b64,
        "qr_debug": qr_debug
    }
