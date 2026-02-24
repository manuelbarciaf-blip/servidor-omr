import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# =========================================
# CONFIGURACIÓN REAL DE TU PLANTILLA A4
# (1 COLUMNA IZQUIERDA + QR ARRIBA)
# =========================================
VALORES_OMR = {
    # Columna izquierda completa de burbujas (ANTES estaba demasiado a la derecha)
    "x0": 0.05,
    "x1": 0.36,

    # Primera burbuja en el cuarto superior (como dijiste)
    "y0": 0.22,
    "y1": 0.88
}

NUM_OPCIONES = 4  # A B C D

# Umbrales calibrados para:
# - Foto móvil
# - Bolígrafo azul/negro
# - Burbuja roja (solo contorno)
UMBRAL_MARCA = 0.18     # marca clara rellena
UMBRAL_DOBLE = 0.75     # doble marca = inválida
UMBRAL_VACIO = 0.04     # pregunta en blanco
UMBRAL_DEBIL = 0.10     # marca débil (revisar)

# =========================================
# LECTURA QR (LA TUYA FUNCIONA - NO TOCO)
# =========================================
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


# =========================================
# NORMALIZACIÓN PRO PARA FOTO DE MÓVIL
# (CLAVE PARA JPG)
# =========================================
def normalizar_imagen(img):
    # Normalizar SIEMPRE a A4 (estabilidad absoluta)
    img_resized = cv2.resize(img, (2480, 3508))

    # Convertir a HSV (mejor que gris para tinta real)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Tinta azul (bolígrafo)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Tinta negra (bolígrafo negro)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combinar máscaras (SOLO tinta, ignora burbuja roja)
    mask_tinta = cv2.bitwise_or(mask_blue, mask_black)

    # Limpiar ruido del móvil
    kernel = np.ones((3, 3), np.uint8)
    mask_tinta = cv2.morphologyEx(mask_tinta, cv2.MORPH_OPEN, kernel)
    mask_tinta = cv2.GaussianBlur(mask_tinta, (5, 5), 0)

    # Threshold suave (NO agresivo para no rellenar toda la burbuja)
    _, th = cv2.threshold(mask_tinta, 35, 255, cv2.THRESH_BINARY)

    return th, img_resized


# =========================================
# DESKEW AUTOMÁTICO (ENDEREZA FOTO MÓVIL)
# =========================================
def corregir_inclinacion(th):
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 1000:
        return th

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    h, w = th.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    rotated = cv2.warpAffine(
        th, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# =========================================
# RECORTE PORCENTAJE (ZONA OMR REAL)
# =========================================
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    return img[
        int(h * valores["y0"]):int(h * valores["y1"]),
        int(w * valores["x0"]):int(w * valores["x1"])
    ]


# =========================================
# DETECTOR DENSIDAD TIPO GRAVIC (PRO)
# 1 columna fija – 20 preguntas
# =========================================
def detectar_respuestas(zona_bin, zona_color, total_preguntas=20):
    filas = total_preguntas
    opciones = NUM_OPCIONES

    h, w = zona_bin.shape

    # IMPORTANTE: ajuste compacto (tu plantilla es densa)
    factor_compacto = 0.96
    alto_fila = int((h * factor_compacto) / filas)
    ancho_op = w // opciones

    letras = ["A", "B", "C", "D"]
    respuestas = []
    mapa = zona_color.copy()

    for fila in range(filas):
        y0 = fila * alto_fila
        y1 = y0 + alto_fila

        if y1 > h:
            respuestas.append(None)
            continue

        fila_img = zona_bin[y0:y1, :]

        densidades = []
        coords = []

        for o in range(opciones):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op

            celda = fila_img[:, x0:x1]

            total_pixeles = celda.size
            tinta = cv2.countNonZero(celda)
            densidad = tinta / total_pixeles if total_pixeles > 0 else 0

            densidades.append(densidad)
            coords.append((x0, y0, x1, y1))

        max_d = max(densidades)
        idx_max = densidades.index(max_d)

        orden = sorted(densidades, reverse=True)
        segundo = orden[1] if len(orden) > 1 else 0

        # VACÍA
        if max_d < UMBRAL_VACIO:
            respuestas.append(None)
            continue

        # DOBLE MARCA (INVÁLIDA)
        if segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
            for (x0, y0, x1, y1) in coords:
                cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 0, 255), 2)
            continue

        # MARCA DÉBIL (REVISAR)
        if max_d < UMBRAL_MARCA:
            respuestas.append("?")
            x0, y0, x1, y1 = coords[idx_max]
            cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 255), 3)
            continue

        # RESPUESTA VÁLIDA
        respuestas.append(letras[idx_max])
        x0, y0, x1, y1 = coords[idx_max]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa


# =========================================
# FUNCIÓN PRINCIPAL (API DOCKER + PHP)
# =========================================
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen no válida"}

    # 1. Leer QR (antes de cualquier transformación)
    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    # 2. Normalización profesional (móvil)
    th, img_norm = normalizar_imagen(img)

    # 3. Deskew automático
    th_corr = corregir_inclinacion(th)

    # 4. Recorte zona OMR real (columna izquierda)
    zona_bin = recortar_porcentual(th_corr, VALORES_OMR)
    zona_color = recortar_porcentual(img_norm, VALORES_OMR)

    # 5. Detectar respuestas (1 columna fija)
    respuestas, mapa = detectar_respuestas(
        zona_bin,
        zona_color,
        total_preguntas=20
    )

    # Debug visual para tu corregir_omr.php
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
        "debug_image": debug_b64,  # hoja recortada (la que NO veías)
        "debug_map": debug_map     # detección pintada
    }
