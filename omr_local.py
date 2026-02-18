#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# 1) LECTURA QR
# ---------------------------------------------------------
def leer_qr(img):
    codes = zbar_decode(img)
    if not codes:
        return None, None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return data, id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# 2) DETECCIÓN DE CUADRADOS (warp)
# ---------------------------------------------------------
def encontrar_cuadrados(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        # Cuadrados pequeños → área baja
        if area < 80 or area > 8000:
            continue

        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        if 0.75 < ratio < 1.25:
            candidates.append((x, y, approx))

    if len(candidates) < 4:
        return None

    # Ordenar por posición
    pts = sorted(candidates, key=lambda p: (p[1], p[0]))

    # Top-left, top-right, bottom-left, bottom-right
    pts_sorted = [
        pts[0][2].reshape(4,2).mean(axis=0),
        pts[1][2].reshape(4,2).mean(axis=0),
        pts[-2][2].reshape(4,2).mean(axis=0),
        pts[-1][2].reshape(4,2).mean(axis=0)
    ]

    return np.float32(pts_sorted)

def warp_hoja(img, corners):
    dst_w, dst_h = 900, 1300
    dst = np.float32([
        [50,50],
        [dst_w-50,50],
        [50,dst_h-50],
        [dst_w-50,dst_h-50]
    ])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (dst_w, dst_h))

# ---------------------------------------------------------
# 3) PLANTILLA 20 PREGUNTAS
# ---------------------------------------------------------
def detectar_respuestas_20(warped):
    h, w = warped.shape[:2]

    y0 = int(h * 0.12)
    y1 = int(h * 0.70)
    x0 = int(w * 0.05)
    x1 = int(w * 0.80)

    zona = warped[y0:y1, x0:x1]

    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    filas = 20
    cols = 4
    respuestas = []

    h_z, w_z = th.shape
    alto_fila = h_z // filas
    ancho_col = w_z // cols

    letras = ["A","B","C","D"]

    for i in range(filas):
        fila = th[i*alto_fila:(i+1)*alto_fila, :]

        valores = []
        for c in range(cols):
            celda = fila[:, c*ancho_col:(c+1)*ancho_col]
            negro = cv2.countNonZero(celda)
            valores.append(negro)

        max_val = max(valores)
        idx = valores.index(max_val)

        if max_val < 40:
            respuestas.append(None)
        else:
            respuestas.append(letras[idx])

    return respuestas

# ---------------------------------------------------------
# 4) PLANTILLA 60 PREGUNTAS (3 columnas × 20 filas)
# ---------------------------------------------------------
def detectar_respuestas_60(warped):
    h, w = warped.shape[:2]

    y0 = int(h * 0.10)
    y1 = int(h * 0.92)
    x0 = int(w * 0.05)
    x1 = int(w * 0.95)

    zona = warped[y0:y1, x0:x1]

    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    filas = 20
    columnas = 3
    opciones = 4

    h_z, w_z = th.shape
    alto_fila = h_z // filas
    ancho_columna = w_z // columnas
    ancho_opcion = ancho_columna // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    for col in range(columnas):
        x_col0 = col * ancho_columna
        x_col1 = (col+1) * ancho_columna

        for fila in range(filas):
            y_f0 = fila * alto_fila
            y_f1 = (fila+1) * alto_fila

            fila_img = th[y_f0:y_f1, x_col0:x_col1]

            valores = []
            for o in range(opciones):
                x_o0 = o * ancho_opcion
                x_o1 = (o+1) * ancho_opcion
                celda = fila_img[:, x_o0:x_o1]
                negro = cv2.countNonZero(celda)
                valores.append(negro)

            max_val = max(valores)
            idx = valores.index(max_val)

            if max_val < 40:
                respuestas.append(None)
            else:
                respuestas.append(letras[idx])

    return respuestas

# ---------------------------------------------------------
# 5) MAIN
# ---------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Uso: omr_local.py <imagen>"}))
        return

    ruta = sys.argv[1]
    img = cv2.imread(ruta)
    if img is None:
        print(json.dumps({"ok": False, "error": "No se pudo leer la imagen"}))
        return

    codigo, id_examen, id_alumno, fecha_qr = leer_qr(img)

    corners = encontrar_cuadrados(img)
    if corners is None:
        warped = img.copy()
        warped_ok = False
    else:
        warped = warp_hoja(img, corners)
        warped_ok = True

    # Selección automática por id_examen
    if id_examen == 272:
        respuestas = detectar_respuestas_60(warped)
    else:
        respuestas = detectar_respuestas_20(warped)

    _, buffer = cv2.imencode(".jpg", warped)
    debug_b64 = base64.b64encode(buffer).decode()

    out = {
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha_qr": fecha_qr,
        "respuestas": respuestas,
        "warp_ok": warped_ok,
        "debug_image": debug_b64
    }
    print(json.dumps(out))

# ---------------------------------------------------------
# 6) CAPTURA GLOBAL DE ERRORES
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(json.dumps({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }))
