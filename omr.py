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
# LECTURA QR ROBUSTA (ANTES DE PROCESAR)
# ---------------------------------------------------------
def leer_qr_original(img):
    qr_img = img.copy()

    # Reducir tamaño para mejorar detección
    qr_img = cv2.resize(qr_img, (900, 1300))

    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    codes = zbar_decode(gray)

    if not codes:
        return None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return id_examen, id_alumno, fecha_qr

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
# RECORTE OMR
# ---------------------------------------------------------
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    return img[
        int(h * valores["y0"]):int(h * valores["y1"]),
        int(w * valores["x0"]):int(w * valores["x1"])
    ]

# ---------------------------------------------------------
# DETECCIÓN DE BURBUJAS + DOBLE MARCA + MARCA DÉBIL + MAPA VISUAL
# ---------------------------------------------------------
def detectar_respuestas_20(zona_bin, zona_color):
    filas = 20
    opciones = 4

    h, w = zona_bin.shape
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A","B","C","D"]
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

        ordenados = sorted(valores, reverse=True)
        max_val = ordenados[0]
        segundo = ordenados[1]
        media = np.mean(valores)

        # 1) Vacía (umbral absoluto)
        if max_val < 150:
            respuestas.append(None)
            for (x0,y0,x1,y1) in coords:
                cv2.rectangle(mapa, (x0,y0), (x1,y1), (255,255,255), 2)
            continue

        # 2) Doble marca
        if segundo > max_val * 0.70:
            respuestas.append("X")
            for (x0,y0,x1,y1) in coords:
                cv2.rectangle(mapa, (x0,y0), (x1,y1), (0,0,255), 2)
            continue

        # 3) Marca débil
        if max_val < media * 1.10:
            respuestas.append("?")
            idx = valores.index(max_val)
            x0,y0,x1,y1 = coords[idx]
            cv2.rectangle(mapa, (x0,y0), (x1,y1), (0,255,255), 3)
            continue

        # 4) Marca clara
        idx = valores.index(max_val)
        respuestas.append(letras[idx])
        x0,y0,x1,y1 = coords[idx]
        cv2.rectangle(mapa, (x0,y0), (x1,y1), (0,255,0), 3)

    return respuestas, mapa
# ---------------------------------------------------------
# PROCESAR IMAGEN COMPLETA
# ---------------------------------------------------------
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "No se pudo decodificar la imagen"}

    # Leer QR en imagen limpia
    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    # Normalizar imagen
    th, img_norm = normalizar_imagen(img)

    # Corregir inclinación
    th_corr = corregir_inclinacion(th)

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
