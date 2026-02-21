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
    "y1": 0.86
}

# ---------------------------------------------------------
# NORMALIZACIÓN DE IMAGEN
# ---------------------------------------------------------
def normalizar_imagen(img):
    img = cv2.resize(img, (2480, 3508))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return th, img

# ---------------------------------------------------------
# CORRECCIÓN DE INCLINACIÓN
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
    return cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ---------------------------------------------------------
# LECTURA QR
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

    return id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# RECORTE OMR
# ---------------------------------------------------------
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    return img[
        int(h * valores["y0"]):int(h * valores["y1"]),
        int(w * valores["x0"]):int(w * valores["x1"])
    ]

# ---------------------------------------------------------
# DETECCIÓN DE BURBUJAS + MAPA VISUAL
# ---------------------------------------------------------
def detectar_respuestas_20(zona_bin, zona_color):
    filas = 20
    opciones = 4

    h, w = zona_bin.shape
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    # Para el mapa visual
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

        ordenados = sorted(valores, reverse=True)
        max_val = ordenados[0]
        segundo = ordenados[1]
        media = np.mean(valores)

        # Caso 1: vacía
        if max_val < media * 1.25:
            respuestas.append(None)
            color = (255,255,255)
            for (x0,y0,x1,y1) in coords:
                cv2.rectangle(mapa, (x0,y0), (x1,y1), color, 2)
            continue

        # Caso 2: marca muy débil
        if max_val < media * 1.60:
            respuestas.append("?")
            idx = valores.index(max_val)
            x0,y0,x1,y1 = coords[idx]
            cv2.rectangle(mapa, (x0,y0), (x1,y1), (0,255,255), 3)  # amarillo
            continue

        # Caso 3: doble marca
        if segundo > max_val * 0.75:
            respuestas.append("X")
            for (x0,y0,x1,y1) in coords:
                cv2.rectangle(mapa, (x0,y0), (x1,y1), (0,0,255), 2)  # rojo
            continue

        # Caso 4: una sola marca clara
        idx = valores.index(max_val)
        respuestas.append(letras[idx])

        x0,y0,x1,y1 = coords[idx]
        cv2.rectangle(mapa, (x0,y0), (x1,y1), (0,255,0), 3)  # verde

    return respuestas, mapa

# ---------------------------------------------------------
# PROCESAR IMAGEN COMPLETA
# ---------------------------------------------------------
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "No se pudo decodificar la imagen"}

    th, img_norm = normalizar_imagen(img)
    th_corr = corregir_inclinacion(th)

    # QR después de normalizar
    id_examen, id_alumno, fecha_qr = leer_qr(img_norm)

    # Recorte OMR
    zona_bin = recortar_porcentual(th_corr, VALORES_OMR)
    zona_color = recortar_porcentual(img_norm, VALORES_OMR)

    # Detectar respuestas + mapa visual
    respuestas, mapa = detectar_respuestas_20(zona_bin, zona_color)

    # Debug burbujas
    _, buffer = cv2.imencode(".jpg", zona_color)
    debug_b64 = base64.b64encode(buffer).decode()

    # Debug mapa visual
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
