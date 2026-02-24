import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# CONFIG ZONA OMR (ajustada a tu plantilla)
# ---------------------------------------------------------
VALORES_OMR = {
    "x0": 0.15,
    "y0": 0.22,
    "x1": 0.90,
    "y1": 0.88
}

# Umbrales tipo Gravic (ajustados para bolígrafo azul/negro)
UMBRAL_VACIO = 0.12
UMBRAL_DEBIL = 0.20
UMBRAL_DOBLE = 0.75

# ---------------------------------------------------------
# LECTURA QR
# ---------------------------------------------------------
def leer_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    codes = zbar_decode(gray)

    if not codes:
        return None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha = partes[2] if len(partes) >= 3 else None

    return id_examen, id_alumno, fecha

# ---------------------------------------------------------
# NORMALIZACIÓN
# ---------------------------------------------------------
def normalizar(img):
    h, w = img.shape[:2]

    if h > 2500:
        escala = 2500 / h
        img = cv2.resize(img, (int(w * escala), 2500))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10
    )

    return th, img

# ---------------------------------------------------------
# DESKEW
# ---------------------------------------------------------
def deskew(th):
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 10:
        return th

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    h, w = th.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    return cv2.warpAffine(
        th,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

# ---------------------------------------------------------
# RECORTE
# ---------------------------------------------------------
def recortar(img, valores):
    h, w = img.shape[:2]

    x0 = int(w * valores["x0"])
    y0 = int(h * valores["y0"])
    x1 = int(w * valores["x1"])
    y1 = int(h * valores["y1"])

    return img[y0:y1, x0:x1]

# ---------------------------------------------------------
# DETECCIÓN POR COLUMNAS DINÁMICA
# ---------------------------------------------------------
def detectar_respuestas(zona_bin, zona_color):
    h, w = zona_bin.shape

    # Determinar columnas según ancho
    if w < 800:
        columnas = 1
        filas_por_col = 20
    elif w < 1400:
        columnas = 2
        filas_por_col = 20
    else:
        columnas = 3
        filas_por_col = 20

    opciones = 4
    ancho_col = w // columnas
    letras = ["A", "B", "C", "D"]

    respuestas = []
    mapa = zona_color.copy()

    for col in range(columnas):
        x_col0 = col * ancho_col
        x_col1 = (col + 1) * ancho_col

        col_bin = zona_bin[:, x_col0:x_col1]
        col_color = mapa[:, x_col0:x_col1]

        alto_fila = h // filas_por_col
        ancho_op = ancho_col // opciones

        for fila in range(filas_por_col):
            y0 = fila * alto_fila
            y1 = (fila + 1) * alto_fila

            fila_bin = col_bin[y0:y1, :]

            densidades = []
            coords = []

            for o in range(opciones):
                x0 = o * ancho_op
                x1 = (o + 1) * ancho_op

                celda = fila_bin[:, x0:x1]
                area = celda.size
                negros = cv2.countNonZero(celda)

                densidad = negros / float(area)
                densidades.append(densidad)
                coords.append((x_col0 + x0, y0, x_col0 + x1, y1))

            max_d = max(densidades)
            idx = densidades.index(max_d)

            sorted_d = sorted(densidades, reverse=True)

            # VACÍA
            if max_d < UMBRAL_VACIO:
                respuestas.append(None)
                continue

            # DÉBIL
            if max_d < UMBRAL_DEBIL:
                respuestas.append("?")
                x0,y0,x1,y1 = coords[idx]
                cv2.rectangle(mapa,(x0,y0),(x1,y1),(0,255,255),3)
                continue

            # DOBLE
            if sorted_d[1] > max_d * UMBRAL_DOBLE:
                respuestas.append("X")
                for i,d in enumerate(densidades):
                    if d > max_d * UMBRAL_DOBLE:
                        x0,y0,x1,y1 = coords[i]
                        cv2.rectangle(mapa,(x0,y0),(x1,y1),(0,0,255),3)
                continue

            # VÁLIDA
            respuestas.append(letras[idx])
            x0,y0,x1,y1 = coords[idx]
            cv2.rectangle(mapa,(x0,y0),(x1,y1),(0,255,0),3)

    return respuestas, mapa

# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def procesar_omr(binario):
    img_array = np.frombuffer(binario, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Imagen inválida"}

    id_examen, id_alumno, fecha = leer_qr(img)

    th, img_norm = normalizar(img)
    th_corr = deskew(th)

    zona_bin = recortar(th_corr, VALORES_OMR)
    zona_color = recortar(img_norm, VALORES_OMR)

    respuestas, mapa = detectar_respuestas(zona_bin, zona_color)

    _, buf1 = cv2.imencode(".jpg", zona_color)
    debug_image = base64.b64encode(buf1).decode()

    _, buf2 = cv2.imencode(".jpg", mapa)
    debug_map = base64.b64encode(buf2).decode()

    return {
        "ok": True,
        "codigo": f"{id_examen}|{id_alumno}|{fecha}" if id_examen else None,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha_qr": fecha,
        "respuestas": respuestas,
        "debug_image": debug_image,
        "debug_map": debug_map
    }
