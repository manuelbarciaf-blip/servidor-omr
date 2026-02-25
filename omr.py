import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# =========================================
# CONFIGURACIÓN A4 CENTRAL
# =========================================
VALORES_OMR = {
    "x0": 0.30,  # zona central
    "x1": 0.70,
    "y0": 0.32,  # tercio superior
    "y1": 0.90
}

NUM_OPCIONES = 4

# Umbrales optimizados para azul sobre círculo negro
UMBRAL_MARCA = 0.12
UMBRAL_DOBLE = 0.70
UMBRAL_VACIO = 0.02
UMBRAL_DEBIL = 0.07


# =========================================
# LECTURA QR
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

    id_examen = int(partes[0]) if partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else None
    fecha_qr = partes[2] if len(partes) > 2 else None

    return id_examen, id_alumno, fecha_qr


# =========================================
# NORMALIZACIÓN SOLO AZUL (IGNORA NEGRO)
# =========================================
def normalizar_imagen(img):
    img_resized = cv2.resize(img, (2480, 3508))

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Detectar solo azul (relleno del alumno)
    lower_blue = np.array([95, 60, 60])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.GaussianBlur(mask_blue, (5, 5), 0)

    _, th = cv2.threshold(mask_blue, 25, 255, cv2.THRESH_BINARY)

    return th, img_resized


# =========================================
# DESKEW
# =========================================
def corregir_inclinacion(th):
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 1000:
        return th

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    h, w = th.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(
        th, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# =========================================
# RECORTE ZONA OMR
# =========================================
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]
    return img[
        int(h * valores["y0"]):int(h * valores["y1"]),
        int(w * valores["x0"]):int(w * valores["x1"])
    ]


# =========================================
# DETECTOR DENSIDAD PROFESIONAL
# =========================================
def detectar_respuestas(zona_bin, zona_color, total_preguntas):
    h, w = zona_bin.shape

    alto_fila = int(h / total_preguntas)
    ancho_op = int(w / NUM_OPCIONES)

    letras = ["A", "B", "C", "D"]
    respuestas = []
    mapa = zona_color.copy()

    for fila in range(total_preguntas):
        y0 = fila * alto_fila
        y1 = y0 + alto_fila

        fila_img = zona_bin[y0:y1, :]

        densidades = []
        coords = []

        for o in range(NUM_OPCIONES):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op

            celda = fila_img[:, x0:x1]

            total = celda.size
            tinta = cv2.countNonZero(celda)
            densidad = tinta / total if total > 0 else 0

            densidades.append(densidad)
            coords.append((x0, y0, x1, y1))

        max_d = max(densidades)
        idx = densidades.index(max_d)

        orden = sorted(densidades, reverse=True)
        segundo = orden[1] if len(orden) > 1 else 0

        if max_d < UMBRAL_VACIO:
            respuestas.append(None)
            continue

        if segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
            for (x0, y0, x1, y1) in coords:
                cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 0, 255), 2)
            continue

        if max_d < UMBRAL_MARCA:
            respuestas.append("?")
            x0, y0, x1, y1 = coords[idx]
            cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 255), 2)
            continue

        respuestas.append(letras[idx])
        x0, y0, x1, y1 = coords[idx]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa


# =========================================
# FUNCIÓN PRINCIPAL
# =========================================
def procesar_omr(binario, total_preguntas):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen no válida"}

    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    th, img_norm = normalizar_imagen(img)
    th_corr = corregir_inclinacion(th)

    zona_bin = recortar_porcentual(th_corr, VALORES_OMR)
    zona_color = recortar_porcentual(img_norm, VALORES_OMR)

    respuestas, mapa = detectar_respuestas(
        zona_bin,
        zona_color,
        total_preguntas
    )

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
