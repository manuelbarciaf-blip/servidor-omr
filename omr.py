import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64
# =========================================
# CONFIGURACIÓN CALIBRADA A PLANTILLA REAL
# =========================================
VALORES_OMR = {
    "x0": 0.14,   # empieza en la columna de números
    "y0": 0.18,
    "x1": 0.32,   # termina justo después de la burbuja D
    "y1": 0.88
}

# Umbrales calibrados para boli azul sobre círculo rojo
UMBRAL_MARCA = 0.45      # marca válida
UMBRAL_DOBLE = 0.75      # doble marca
UMBRAL_VACIO = 0.10      # realmente en blanco
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
    # A4 real para estabilizar proporciones
    img_resized = cv2.resize(img, (2480, 3508))

    # Convertir a escala de grises
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Eliminar sombras del móvil (CLAVE)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    div = cv2.divide(gray, blur, scale=255)

    # Binarización OMR real (mejor que adaptive para hojas)
    _, th = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
