import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64
# =========================================
# CONFIGURACIÓN CALIBRADA A PLANTILLA REAL
# =========================================
# CONFIGURACIÓN OMR PROFESIONAL (MÓVIL + AZUL/NEGRO + BURBUJA ROJA)
VALORES_OMR = {
    "x0": 0.16,   # zona real de respuestas (ajustada a tu hoja)
    "y0": 0.18,
    "x1": 0.34,
    "y1": 0.90
}

# Umbrales calibrados para foto móvil
UMBRAL_MARCA = 0.30     # marca válida (azul o negro)
UMBRAL_DOBLE = 0.70     # doble marca = inválida (X)
UMBRAL_VACIO = 0.07     # pregunta en blanco
# =========================================
# LECTURA QR (NO SE TOCA - YA TE FUNCIONA)
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
# NORMALIZACIÓN ROBUSTA PARA FOTO DE MÓVIL
# =========================================
def normalizar_imagen(img):
    # 1. Redimensionar a tamaño A4 estándar (muy importante para estabilidad)
    img_resized = cv2.resize(img, (2480, 3508))

    # 2. Convertir a HSV (mejor para detectar tinta que escala de grises)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # 3. Detectar tinta AZUL (bolígrafo)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4. Detectar tinta NEGRA (bolígrafo negro)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 5. Combinar ambas máscaras (azul + negro)
    mask_tinta = cv2.bitwise_or(mask_blue, mask_black)

    # 6. Limpiar ruido (círculos rojos y sombras del móvil)
    kernel = np.ones((3, 3), np.uint8)
    mask_tinta = cv2.morphologyEx(mask_tinta, cv2.MORPH_OPEN, kernel)
    mask_tinta = cv2.GaussianBlur(mask_tinta, (5, 5), 0)

    # 7. Threshold final suave (NO agresivo para no pintar toda la burbuja)
    _, th = cv2.threshold(mask_tinta, 40, 255, cv2.THRESH_BINARY)

    return th, img_resized
# =========================================
# DESKEW AUTOMÁTICO (ENDEREZAR FOTO)
# =========================================
def corregir_inclinacion(th):
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 500:
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
# RECORTE INTELIGENTE ZONA OMR
# =========================================
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    return img[
        int(h * valores["y0"]):int(h * valores["y1"]),
        int(w * valores["x0"]):int(w * valores["x1"])
    ]

# =========================================
# DETECTOR DE DENSIDAD TIPO GRAVIC (PRO)
# =========================================
def detectar_respuestas(zona_bin, zona_color, total_preguntas=20):
    filas = total_preguntas
    opciones = 4

    h, w = zona_bin.shape
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A", "B", "C", "D"]
    respuestas = []
    mapa = zona_color.copy()

    for fila in range(filas):
        y0 = fila * alto_fila
        y1 = (fila + 1) * alto_fila

        fila_img = zona_bin[y0:y1, :]

        densidades = []
        coords = []

        for o in range(opciones):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op

            celda = fila_img[:, x0:x1]

            # DENSIDAD REAL (clave profesional)
            total_pixeles = celda.size
            negros = cv2.countNonZero(celda)
            densidad = negros / total_pixeles

            densidades.append(densidad)
            coords.append((x0, y0, x1, y1))

        max_d = max(densidades)
        idx_max = densidades.index(max_d)

        # ORDENAR PARA DETECTAR DOBLE MARCA
        orden = sorted(densidades, reverse=True)
        segundo = orden[1]

        # ---------------------------
        # PREGUNTA EN BLANCO
        # ---------------------------
        if max_d < UMBRAL_VACIO:
            respuestas.append(None)
            continue

        # ---------------------------
        # DOBLE MARCA (INVÁLIDA)
        # ---------------------------
        if segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
            for (x0, y0, x1, y1) in coords:
                cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 0, 255), 2)
            continue

        # ---------------------------
        # MARCA DÉBIL (REVISAR)
        # ---------------------------
        if max_d < UMBRAL_MARCA:
            respuestas.append("?")
            x0, y0, x1, y1 = coords[idx_max]
            cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 255), 3)
            continue

        # ---------------------------
        # RESPUESTA VÁLIDA
        # ---------------------------
        respuestas.append(letras[idx_max])
        x0, y0, x1, y1 = coords[idx_max]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa

# =========================================
# FUNCIÓN PRINCIPAL (API)
# =========================================
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen no válida"}

    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    th, img_norm = normalizar_imagen(img)
    th_corr = corregir_inclinacion(th)

    zona_bin = recortar_porcentual(th_corr, VALORES_OMR)
    zona_color = recortar_porcentual(img_norm, VALORES_OMR)

    respuestas, mapa = detectar_respuestas(zona_bin, zona_color, total_preguntas=20)

    # Debug visual (lo que tu PHP muestra)
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
