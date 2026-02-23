import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

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
# NORMALIZACIÓN DE IMAGEN (A4 ESTÁNDAR)
# ---------------------------------------------------------
def normalizar_imagen(img):
    img_resized = cv2.resize(img, (2480, 3508))  # A4 300dpi
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )
    return th, img_resized

# ---------------------------------------------------------
# DETECTAR LOS 4 CUADRADOS NEGROS DE REFERENCIA OMR
# ---------------------------------------------------------
def detectar_cuadrados_referencia(th):
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuadrados = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:  # filtrar ruido
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)

        # Buscamos cuadrados grandes (como los de tu plantilla)
        if 0.7 < ratio < 1.3:
            cuadrados.append((x, y, w, h, area))

    # Ordenar por área (los 4 más grandes serán las esquinas)
    cuadrados = sorted(cuadrados, key=lambda c: c[4], reverse=True)[:4]

    if len(cuadrados) < 4:
        return None

    # Obtener centros
    centros = []
    for (x, y, w, h, _) in cuadrados:
        cx = x + w // 2
        cy = y + h // 2
        centros.append([cx, cy])

    centros = np.array(centros, dtype="float32")

    # Ordenar: arriba-izq, arriba-der, abajo-izq, abajo-der
    s = centros.sum(axis=1)
    diff = np.diff(centros, axis=1)

    top_left = centros[np.argmin(s)]
    bottom_right = centros[np.argmax(s)]
    top_right = centros[np.argmin(diff)]
    bottom_left = centros[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")

# ---------------------------------------------------------
# CORREGIR PERSPECTIVA USANDO LOS CUADRADOS OMR
# ---------------------------------------------------------
def corregir_perspectiva(img_color, th):
    puntos = detectar_cuadrados_referencia(th)

    if puntos is None:
        # fallback si no detecta cuadrados
        return img_color, th

    (tl, tr, bl, br) = puntos

    ancho = int(max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl)
    ))

    alto = int(max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr)
    ))

    destino = np.array([
        [0, 0],
        [ancho - 1, 0],
        [0, alto - 1],
        [ancho - 1, alto - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(puntos, destino)

    warped_color = cv2.warpPerspective(img_color, M, (ancho, alto))
    warped_th = cv2.warpPerspective(th, M, (ancho, alto))

    return warped_color, warped_th

# ---------------------------------------------------------
# RECORTE AUTOMÁTICO DEL ÁREA DE BURBUJAS (SIN PORCENTAJES)
# ---------------------------------------------------------
def recortar_area_burbujas(img_color, th):
    h, w = th.shape

    # Quitamos márgenes donde están los cuadrados y el QR
    margen_x = int(w * 0.12)
    margen_y_top = int(h * 0.18)
    margen_y_bottom = int(h * 0.08)

    zona_color = img_color[
        margen_y_top:h - margen_y_bottom,
        margen_x:w - margen_x
    ]

    zona_bin = th[
        margen_y_top:h - margen_y_bottom,
        margen_x:w - margen_x
    ]

    return zona_color, zona_bin

# ---------------------------------------------------------
# DETECCIÓN DE RESPUESTAS (DINÁMICA)
# ---------------------------------------------------------
def detectar_respuestas(zona_bin, zona_color, filas=20, opciones=4):
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

        if max_val < 40:
            respuestas.append(None)
            continue

        respuestas.append(letras[idx])

        x0, y0, x1, y1 = coords[idx]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa

# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL (USADA POR FASTAPI)
# ---------------------------------------------------------
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "No se pudo decodificar la imagen"}

    # 1. Leer QR
    id_examen, id_alumno, fecha_qr = leer_qr_original(img)

    # 2. Normalizar imagen
    th, img_norm = normalizar_imagen(img)

    # 3. Corregir perspectiva usando los cuadrados OMR
    warped_color, warped_th = corregir_perspectiva(img_norm, th)

    # 4. Recortar automáticamente el área real de burbujas
    zona_color, zona_bin = recortar_area_burbujas(warped_color, warped_th)

    # 5. Detectar respuestas (20 por defecto, luego lo hacemos dinámico si quieres)
    respuestas, mapa = detectar_respuestas(zona_bin, zona_color, filas=20)

    # Debug imágenes
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
