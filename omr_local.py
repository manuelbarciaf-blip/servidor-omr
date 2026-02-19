#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64

# ---------------------------------------------------------
# 1) LECTURA QR (ROBUSTA)
# ---------------------------------------------------------
def leer_qr(img):
    codes = zbar_decode(img)

    if not codes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        codes = zbar_decode(gray)

    if not codes:
        return None, None, None, None

    data = codes[0].data.decode("utf-8").strip()
    partes = data.split("|")

    id_examen = int(partes[0]) if len(partes) >= 1 and partes[0].isdigit() else None
    id_alumno = int(partes[1]) if len(partes) >= 2 and partes[1].isdigit() else None
    fecha_qr  = partes[2] if len(partes) >= 3 else None

    return data, id_examen, id_alumno, fecha_qr

# ---------------------------------------------------------
# 2) DETECTAR 4 ESQUINAS NEGRAS
# ---------------------------------------------------------
def encontrar_cuadrados(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    h, w = gray.shape
    area_img = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_img * 0.0005 or area > area_img * 0.02:
            continue

        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(approx)
            ratio = bw / float(bh)
            if 0.7 < ratio < 1.3:
                cx = x + bw / 2
                cy = y + bh / 2
                candidatos.append((cx, cy, approx))

    if len(candidatos) < 4:
        return None

    candidatos = sorted(candidatos, key=lambda p: p[1])
    top = sorted(candidatos[:2], key=lambda p: p[0])
    bottom = sorted(candidatos[-2:], key=lambda p: p[0])

    def centro(a):
        return a.reshape(4,2).mean(axis=0)

    return np.float32([
        centro(top[0][2]),
        centro(top[1][2]),
        centro(bottom[0][2]),
        centro(bottom[1][2])
    ])

# ---------------------------------------------------------
# 3) WARP
# ---------------------------------------------------------
def warp_hoja(img, corners):
    dst_w, dst_h = 1000, 1400
    dst = np.float32([
        [80, 80],
        [dst_w-80, 80],
        [80, dst_h-80],
        [dst_w-80, dst_h-80]
    ])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (dst_w, dst_h))

# ---------------------------------------------------------
# 4) DETECCIÓN DE 20 PREGUNTAS (TU PLANTILLA REAL)
# ---------------------------------------------------------
def detectar_respuestas_20(warped):
    h, w = warped.shape[:2]

    # Recorte calibrado para tu hoja
    x0 = int(w * 0.12)
    y0 = int(h * 0.23)
    x1 = int(w * 0.88)
    y1 = int(h * 0.87)

    zona = warped[y0:y1, x0:x1]

    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    filas = 20
    opciones = 4

    h_z, w_z = th.shape
    alto_fila = h_z // filas
    ancho_op = w_z // opciones

    letras = ["A","B","C","D"]
    respuestas = []

    for fila in range(filas):
        y0f = fila * alto_fila
        y1f = (fila + 1) * alto_fila
        fila_img = th[y0f:y1f, :]

        valores = []
        for o in range(opciones):
            x0o = o * ancho_op
            x1o = (o + 1) * ancho_op
            celda = fila_img[:, x0o:x1o]

            ch, cw = celda.shape
            cy0 = int(ch * 0.25)
            cy1 = int(ch * 0.75)
            cx0 = int(cw * 0.25)
            cx1 = int(cw * 0.75)

            centro = celda[cy0:cy1, cx0:cx1]
            negro = cv2.countNonZero(centro)
            valores.append(negro)

        max_val = max(valores)
        idx = valores.index(max_val)
        media = np.mean(valores)

        if max_val < media * 1.6:
            respuestas.append(None)
        else:
            respuestas.append(letras[idx])

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

        codigo, id_examen, id_alumno, fecha_qr = leer_qr(img)

        corners = encontrar_cuadrados(img)

        if corners is not None:
            warped = warp_hoja(img, corners)
            warp_ok = True
        else:
            warped = img.copy()
            warp_ok = False

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
            "warp_ok": warp_ok,
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
