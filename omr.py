import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# =========================================
# CONFIGURACIÓN PLANTILLA REAL (A4 CENTRAL)
# - QR: arriba derecha
# - Burbujas: columna central
# - Primera burbuja: tercio superior
# =========================================
VALORES_OMR = {
    "x0": 0.32,   # columna central (nuevo diseño)
    "x1": 0.68,
    "y0": 0.22,   # empiezan en tercio superior
    "y1": 0.92
}

NUM_OPCIONES = 4  # A B C D

# Umbrales calibrados para:
# - Foto móvil JPG
# - Burbujas negras
# - Bolígrafo azul o negro (relleno completo)
UMBRAL_MARCA = 0.22     # marca clara válida
UMBRAL_DOBLE = 0.75     # doble marca = inválida
UMBRAL_VACIO = 0.05     # pregunta en blanco
UMBRAL_DEBIL = 0.12     # marca débil (revisar)

# =========================================
# LECTURA QR ULTRA ROBUSTA (2x2 cm + móvil)
# =========================================
def leer_qr_original(img):
    # 1. Intento directo (imagen original SIN tocar)
    codes = zbar_decode(img)
    if codes:
        return parse_qr(codes[0].data.decode("utf-8").strip())

    h, w = img.shape[:2]

    # 2. Recorte zona superior derecha (donde está tu QR)
    zona_qr = img[int(0.0*h):int(0.35*h), int(0.55*w):int(1.0*w)]
    codes = zbar_decode(zona_qr)
    if codes:
        return parse_qr(codes[0].data.decode("utf-8").strip())

    # 3. Escala de grises con contraste (clave en JPG móvil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    codes = zbar_decode(gray)
    if codes:
        return parse_qr(codes[0].data.decode("utf-8").strip())

    return None, None, None


def parse_qr(data):
    partes = data.split("|")
    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr = partes[2] if len(partes) >= 3 else None
    return id_examen, id_alumno, fecha_qr


# =========================================
# NORMALIZACIÓN PROFESIONAL (MÓVIL + BOLI)
# Detecta tinta azul/negra ignorando burbuja
# =========================================
def normalizar_imagen(img):
    # Normalizar siempre a A4 (estabilidad total)
    img_resized = cv2.resize(img, (2480, 3508))

    # Convertir a HSV (mejor para tinta real)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Tinta azul (bolígrafo)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Tinta negra (bolígrafo negro)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combinar máscaras de tinta (IGNORA el círculo negro impreso)
    mask_tinta = cv2.bitwise_or(mask_blue, mask_black)

    # Limpieza de ruido (sombras móvil)
    kernel = np.ones((3, 3), np.uint8)
    mask_tinta = cv2.morphologyEx(mask_tinta, cv2.MORPH_OPEN, kernel)
    mask_tinta = cv2.GaussianBlur(mask_tinta, (5, 5), 0)

    # Threshold suave (clave para no rellenar burbuja completa)
    _, th = cv2.threshold(mask_tinta, 30, 255, cv2.THRESH_BINARY)

    return th, img_resized


# =========================================
# DESKEW AUTOMÁTICO (ENDEREZAR FOTO)
# =========================================
def corregir_inclinacion(th):
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 2000:
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
# RECORTE ZONA OMR (COLUMNA CENTRAL A4)
# =========================================
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    return img[
        int(h * valores["y0"]):int(h * valores["y1"]),
        int(w * valores["x0"]):int(w * valores["x1"])
    ]


# =========================================
# OMR TIPO GRAVIC (DENSIDAD REAL)
# - Vacío = None
# - Débil = ?
# - Doble = X (inválida)
# =========================================
def detectar_respuestas(zona_bin, zona_color, total_preguntas=20):
    filas = total_preguntas
    opciones = NUM_OPCIONES

    h, w = zona_bin.shape

    # Ajuste compacto profesional
    alto_fila = int((h * 0.97) / filas)
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
# FUNCIÓN PRINCIPAL (COMPATIBLE DOCKER + PHP)
# =========================================
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen no válida"}

    # 🔥 CORRECCIÓN CRÍTICA PARA FOTOS DE MÓVIL (rotación)
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # 1. LEER QR (ANTES DE TODO)
    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    # Si no hay QR, seguimos pero avisamos
    if id_examen is None:
        print("⚠️ QR no detectado")

    # 2. Normalización profesional
    th, img_norm = normalizar_imagen(img)

    # 3. Deskew automático
    th_corr = corregir_inclinacion(th)

    # 4. Recorte zona OMR (columna central)
    zona_bin = recortar_porcentual(th_corr, VALORES_OMR)
    zona_color = recortar_porcentual(img_norm, VALORES_OMR)

    # 5. Número de preguntas (puedes ajustarlo dinámicamente por examen)
    total_preguntas = 20

    # 6. Detección tipo Gravic
    respuestas, mapa = detectar_respuestas(
        zona_bin,
        zona_color,
        total_preguntas=total_preguntas
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
        "debug_image": debug_b64,
        "debug_map": debug_map
    }
