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
# 2) DETECCIÓN DE LOS 4 CUADRADOS (para warp)
# ---------------------------------------------------------
def encontrar_cuadrados(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    h, w = gray.shape
    area_img = w * h

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_img * 0.001:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(approx)
            ratio = bw / float(bh)
            if 0.8 < ratio < 1.2:
                candidates.append((x, y, bw, bh, approx))

    if len(candidates) < 4:
        return None

    pts = []
    for x, y, bw, bh, approx in candidates:
        cx = x + bw/2
        cy = y + bh/2
        pts.append((cx, cy, approx))

    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:4], key=lambda p: p[0])

    def centro(a):
        return a.reshape(4,2).mean(axis=0)

    return np.float32([centro(top[0][2]), centro(top[1][2]),
                       centro(bottom[0][2]), centro(bottom[1][2])])

def warp_hoja(img, corners):
    dst_w, dst_h = 800, 1100
    dst = np.float32([[50,50],[dst_w-50,50],[50,dst_h-50],[dst_w-50,dst_h-50]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (dst_w, dst_h))

# ---------------------------------------------------------
# 3) PLANTILLA 20 PREGUNTAS: ZONA DE BURBUJAS
# ---------------------------------------------------------
def detectar_respuestas_20(warped):
    h, w = warped.shape[:2]

    # Zona aproximada donde está la tabla A–D (ajustable)
    y0 = int(h * 0.25)
    y1 = int(h * 0.90)
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)

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
        y_f0 = i * alto_fila
        y_f1 = (i+1) * alto_fila
        fila = th[y_f0:y_f1, :]

        valores = []
        for c in range(cols):
            x_c0 = c * ancho_col
            x_c1 = (c+1) * ancho_col
            celda = fila[:, x_c0:x_c1]
            negro = cv2.countNonZero(celda)
            valores.append(negro)

        max_val = max(valores)
        idx = valores.index(max_val)

        if max_val < 80:
            respuestas.append(None)
        else:
            respuestas.append(letras[idx])

    return respuestas

# ---------------------------------------------------------
# 4) MAIN
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
# 5) CAPTURA GLOBAL DE ERRORES
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
