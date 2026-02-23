import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# ZONA OMR REAL (AJUSTADA A TU PLANTILLA CON 4 BURBUJAS A-D)
# Foto: SOLO hoja A4 (como indicaste)
# ---------------------------------------------------------
VALORES_OMR = {
    "x0": 0.18,   # antes 0.10 (muy a la izquierda)
    "y0": 0.20,   # debajo del QR y cabecera
    "x1": 0.88,   # incluir A B C D + columnas
    "y1": 0.85    # toda la zona de preguntas
}

# ---------------------------------------------------------
# LECTURA ROBUSTA DEL QR
# ---------------------------------------------------------
def leer_qr_original(img):
    qr_img = cv2.resize(img.copy(), (900, 1300))
    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    codes = zbar_decode(gray)
    if not codes:
        return None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr = partes[2] if len(partes) >= 3 else None

    return id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# NORMALIZACIÓN (OPTIMIZADA PARA FOTO DE MÓVIL DE UNA HOJA)
# ---------------------------------------------------------
def normalizar_imagen(img):
    # NO forzar tamaño A4 (esto rompía tu detección)
    h, w = img.shape[:2]

    # Escalado suave solo si la imagen es enorme
    if h > 2000:
        escala = 2000 / h
        img = cv2.resize(img, (int(w * escala), 2000))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    return th, img

# ---------------------------------------------------------
# CORRECCIÓN DE INCLINACIÓN
# ---------------------------------------------------------
def corregir_inclinacion(th):
    coords = np.column_stack(np.where(th > 0))
    if coords.shape[0] == 0:
        return th

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    h, w = th.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(
        th,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

# ---------------------------------------------------------
# RECORTE PORCENTUAL (YA CORREGIDO PARA TU PLANTILLA)
# ---------------------------------------------------------
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]

    x0 = int(w * valores["x0"])
    y0 = int(h * valores["y0"])
    x1 = int(w * valores["x1"])
    y1 = int(h * valores["y1"])

    # Seguridad: evitar recortes inválidos
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)

    return img[y0:y1, x0:x1]

# ---------------------------------------------------------
# DETECCIÓN DINÁMICA (FUNCIONA DE 20 A 60 PREGUNTAS)
# ---------------------------------------------------------
def detectar_respuestas_dinamico(zona_bin, zona_color):
    h, w = zona_bin.shape

    # Detectar automáticamente número de filas
    # (Optimizado para tus exámenes de 10 en 10)
    if h < 600:
        filas = 20
    elif h < 1000:
        filas = 40
    else:
        filas = 60

    opciones = 4
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A", "B", "C", "D"]
    respuestas = []
    mapa = zona_color.copy()

    for fila in range(filas):
        y0 = fila * alto_fila
        y1 = (fila + 1) * alto_fila
        fila_img = zona_bin[y0:y1, :]

        valores = []
        coords = []

        for o in range(opciones):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op
            celda = fila_img[:, x0:x1]

            negros = cv2.countNonZero(celda)
            valores.append(negros)
            coords.append((x0, y0, x1, y1))

        max_val = max(valores)
        idx = valores.index(max_val)

        # Vacía
        if max_val < 40:
            respuestas.append(None)
            continue

        # Marca clara
        respuestas.append(letras[idx])
        x0, y0, x1, y1 = coords[idx]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa

# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL (COMPATIBLE CON TU main.py y PHP)
# ---------------------------------------------------------
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "No se pudo decodificar la imagen"}

    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    th, img_norm = normalizar_imagen(img)
    th_corr = corregir_inclinacion(th)

    zona_bin = recortar_porcentual(th_corr, VALORES_OMR)
    zona_color = recortar_porcentual(img_norm, VALORES_OMR)

    # DEBUG CRÍTICO (para que vuelvas a ver la imagen en PHP)
    if zona_color.size == 0:
        return {
            "ok": False,
            "error": "Zona OMR vacía (recorte incorrecto)"
        }

    respuestas, mapa = detectar_respuestas_dinamico(zona_bin, zona_color)

    _, buffer = cv2.imencode(".jpg", zona_color)
    debug_b64 = base64.b64encode(buffer).decode()

    _, map_buf = cv2.imencode(".jpg", mapa)
    debug_map = base64.b64encode(map_buf).decode()

    return {
        "ok": True,
        "codigo": f"{id_examen}|{id_alumno}|{fecha_qr}" if id_examen else None,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha_qr": fecha_qr,
        "respuestas": respuestas,
        "debug_image": debug_b64,
        "debug_map": debug_map
    }
