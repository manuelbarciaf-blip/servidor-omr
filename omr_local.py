#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# VALORES OMR DEFINIDOS POR MANUEL (FUNCIONAN EN SU PLANTILLA)
# ---------------------------------------------------------
VALORES_OMR = {
    "x0": 0.25,
    "y0": 0.26,
    "x1": 0.40,
    "y1": 0.77
}

# ---------------------------------------------------------
# 1) RECORTE DEDICADO DEL QR (OPCIÓN B)
# ---------------------------------------------------------
def recorte_qr(img):
    h, w = img.shape[:2]

    # Zona superior derecha donde SIEMPRE está el QR
    x0 = int(w * 0.70)
    y0 = int(h * 0.10)
    x1 = int(w * 0.90)
    y1 = int(h * 0.25)

    return img[y0:y1, x0:x1]

# ---------------------------------------------------------
# 2) LECTURA QR MEJORADA
# ---------------------------------------------------------
def leer_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    codes = zbar_decode(blur)
    if not codes:
        return None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# 3) RECORTE PORCENTUAL DE BURBUJAS (MISMO QUE omr_debur.php)
# ---------------------------------------------------------
def recortar_porcentual(img, valores):
    h, w = img.shape[:2]

    X0 = int(w * valores["x0"])
    Y0 = int(h * valores["y0"])
    X1 = int(w * valores["x1"])
    Y1 = int(h * valores["y1"])

    return img[Y0:Y1, X0:X1]

# ---------------------------------------------------------
# 4) DETECCIÓN DE 20 PREGUNTAS
# ---------------------------------------------------------
def detectar_respuestas_20(zona):
    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    filas = 20
    opciones = 4

    h, w = th.shape
    alto_fila = h // filas
    ancho_op = w // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    for fila in range(filas):
        y0 = fila * alto_fila
        y1 = (fila + 1) * alto_fila
        fila_img = th[y0:y1, :]

        valores = []
        for o in range(opciones):
            x0 = o * ancho_op
            x1 = (o + 1) * ancho_op
            celda = fila_img[:, x0:x1]

            negro = cv2.countNonZero(celda)
            valores.append(negro)

        max_val = max(valores)
        media = np.mean(valores)

        if max_val < media * 1.5:
            respuestas.append(None)
        else:
            respuestas.append(letras[valores.index(max_val)])

    return respuestas

# ---------------------------------------------------------
# 5) MAIN
# ---------------------------------------------------------
def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"ok": False, "error": "Uso: omr_local.py <imagen>"}))
            return

        ruta = sys.argv[1]
        img = cv2.imread(ruta)

        if img is None:
            print(json.dumps({"ok": False, "error": "No se pudo leer la imagen"}))
            return

        # 1) Recortar zona del QR
        qr_zone = recorte_qr(img)

        # 2) Leer QR SOLO en esa zona
        id_examen, id_alumno, fecha_qr = leer_qr(qr_zone)

        # 3) Recortar zona de burbujas
        zona = recortar_porcentual(img, VALORES_OMR)

        # 4) Detectar respuestas
        respuestas = detectar_respuestas_20(zona)

        # 5) Imagen debug
        _, buffer = cv2.imencode(".jpg", zona)
        debug_b64 = base64.b64encode(buffer).decode()

        out = {
            "ok": True,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha_qr": fecha_qr,
            "respuestas": respuestas,
            "debug_image": debug_b64
        }

        print(json.dumps(out))

    except Exception as e:
        import traceback
        print(json.dumps({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }))

if __name__ == "__main__":
    main()
