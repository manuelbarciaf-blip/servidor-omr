#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
import pymysql
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# LEER QR
# ---------------------------------------------------------
def leer_qr(img):
    codes = zbar_decode(img)
    if not codes:
        return None, None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if partes[0].isdigit() else None
    id_alumno = int(partes[1]) if partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return data, id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# OBTENER FORMATO DESDE MYSQL
# ---------------------------------------------------------
def obtener_formato(id_examen):
    conn = pymysql.connect(
        host="localhost",
        user="TU_USUARIO",
        password="TU_PASSWORD",
        database="TU_BD",
        charset="utf8mb4"
    )
    cur = conn.cursor(pymysql.cursors.DictCursor)

    # 1) Obtener número de preguntas del examen
    cur.execute("SELECT num_preguntas FROM examenes WHERE id_examen=%s", (id_examen,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    preguntas = row["num_preguntas"]

    # 2) Obtener formato por número de preguntas
    cur.execute("SELECT * FROM omr_formatos WHERE preguntas=%s", (preguntas,))
    formato = cur.fetchone()

    conn.close()
    return formato

# ---------------------------------------------------------
# DETECTAR CUADRADOS
# ---------------------------------------------------------
def encontrar_cuadrados(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Invertir para que los cuadrados negros se vuelvan blancos
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        # Aceptar cuadrados MUCHO más pequeños y más grandes
        if area < 5 or area > 20000:
            continue

        # Aproximación más permisiva
        approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)

        # Aceptar deformaciones por perspectiva
        if 0.50 < ratio < 1.50:
            candidates.append((x, y, approx))

    if len(candidates) < 4:
        return None

    # Ordenar por posición (arriba-izquierda, arriba-derecha, abajo-izquierda, abajo-derecha)
    pts = sorted(candidates, key=lambda p: (p[1], p[0]))

    pts_sorted = [
        pts[0][2].reshape(4,2).mean(axis=0),
        pts[1][2].reshape(4,2).mean(axis=0),
        pts[-2][2].reshape(4,2).mean(axis=0),
        pts[-1][2].reshape(4,2).mean(axis=0)
    ]

    return np.float32(pts_sorted)
# ---------------------------------------------------------
# DETECTAR BURBUJAS SEGÚN FORMATO
# ---------------------------------------------------------
def detectar_respuestas(warped, formato):
    h, w = warped.shape[:2]

    x0 = int(w * formato["x0"])
    y0 = int(h * formato["y0"])
    x1 = int(w * formato["x1"])
    y1 = int(h * formato["y1"])

    zona = warped[y0:y1, x0:x1]

    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    filas = formato["preguntas"]
    columnas = formato["columnas"]
    opciones = formato["opciones"]

    h_z, w_z = th.shape
    alto_fila = h_z // filas
    ancho_columna = w_z // columnas
    ancho_opcion = ancho_columna // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    for col in range(columnas):
        x_col0 = col * ancho_columna

        for fila in range(filas // columnas):
            y_f0 = fila * alto_fila

            fila_img = th[y_f0:y_f0+alto_fila, x_col0:x_col0+ancho_columna]

            valores = []
            for o in range(opciones):
                x_o0 = o * ancho_opcion
                celda = fila_img[:, x_o0:x_o0+ancho_opcion]
                negro = cv2.countNonZero(celda)
                valores.append(negro)

            max_val = max(valores)
            idx = valores.index(max_val)

            if max_val < 20:
                respuestas.append(None)
            else:
                respuestas.append(letras[idx])

    return respuestas

# ---------------------------------------------------------
# MAIN
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

    formato = obtener_formato(id_examen)
    if formato is None:
        print(json.dumps({"ok": False, "error": "No existe formato para este número de preguntas"}))
        return

    corners = encontrar_cuadrados(img)
    if corners is None:
        warped = img.copy()
        warped_ok = False
    else:
        warped = warp_hoja(img, corners)
        warped_ok = True

    respuestas = detectar_respuestas(warped, formato)

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

if __name__ == "__main__":
    main()
