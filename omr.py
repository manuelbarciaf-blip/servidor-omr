import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# VALORES OMR DEFINIDOS POR MANUEL
# ---------------------------------------------------------
VALORES_OMR = {
    "x0": 0.14,
    "y0": 0.17,
    "x1": 0.74,
    "y1": 0.86,
}

# ---------------------------------------------------------
# NORMALIZACIÓN DE IMAGEN (MEJORA CRÍTICA)
# ---------------------------------------------------------
def normalizar_imagen(img):
    # Escalar a tamaño estándar (A4 aprox)
    img = cv2.resize(img, (2480, 3508))

    # Convertir a gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aumentar contraste
    gray = cv2.equalizeHist(gray)

    # Suavizar ruido
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Binarización adaptativa (mucho mejor que Otsu solo)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

    return th, img

# ---------------------------------------------------------
# CORRECCIÓN DE INCLINACIÓN (DESKEW)
# ---------------------------------------------------------
def corregir_inclinacion(th):
    coords = np.column_stack(np.where(th > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = th.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    th_corr = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return th_corr

# ---------------------------------------------------------
# LECTURA QR EN TODA LA IMAGEN
# ---------------------------------------------------------
def leer_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    codes = zbar_decode(blur)
    if not codes:
        return None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return id_examen, id_alumno, fecha_qr = leer_qr(cv2.cvtColor(th_corr, cv2.COLOR_GRAY2BGR))

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
# DETECCIÓN ROBUSTA DE 20 PREGUNTAS
# ---------------------------------------------------------
def detectar_respuestas_20(zona_bin):
    filas = 20
    opciones = 4

    h, w = zona_bin.shape
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    for fila in range(filas):
        y0 = fila * alto_fila
        y1 = (fila + 1) * alto_fila
        fila_img = zona_bin[y0:y1, :]

        valores = []
        for o in range(opciones):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op
            celda = fila_img[:, x0:x1]

            # Contar píxeles negros
            negros = cv2.countNonZero(celda)
            valores.append(negros)

        max_val = max(valores)
        media = np.mean(valores)

        # Mucho más tolerante
        if max_val < media * 1.25:
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

    # Normalizar imagen
    th, img_norm = normalizar_imagen(img)

    # Corregir inclinación
    th_corr = corregir_inclinacion(th)

    # Leer QR DESPUÉS de normalizar
    id_examen, id_alumno, fecha_qr = leer_qr(img_norm)

    # Recorte de burbujas
    zona = recortar_porcentual(th_corr, VALORES_OMR)

    # Detectar respuestas
    respuestas = detectar_respuestas_20(zona)

    # Debug
    _, buffer = cv2.imencode(".jpg", zona)
    debug_b64 = base64.b64encode(buffer).decode()

    return {
        "ok": True,
        "codigo": f"{id_examen}|{id_alumno}|{fecha_qr}" if id_examen else None,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha_qr": fecha_qr,
        "respuestas": respuestas,
        "debug_image": debug_b64
    }
