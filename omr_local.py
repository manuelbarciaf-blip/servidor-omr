#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
import base64
import os

CONFIG_FILE = "omr_config.json"
drawing = False
ix, iy = -1, -1
rx0, ry0, rx1, ry1 = 0, 0, 0, 0

# ---------------------------------------------------------
# 1) LECTURA QR
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
# 2) DETECTAR CUADRADOS
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
# 4) SELECCIONAR AREA CON RATON (CALIBRACIÓN)
# ---------------------------------------------------------
def seleccionar_area(imagen):
    global ix, iy, drawing, rx0, ry0, rx1, ry1

    clone = imagen.copy()

    def mouse(event, x, y, flags, param):
        global ix, iy, drawing, rx0, ry0, rx1, ry1

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_temp = clone.copy()
                cv2.rectangle(img_temp, (ix, iy), (x, y), (0,255,0), 2)
                cv2.imshow("Calibrar OMR - Selecciona burbujas", img_temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rx0, ry0 = ix, iy
            rx1, ry1 = x, y
            cv2.rectangle(clone, (rx0, ry0), (rx1, ry1), (0,255,0), 2)
            cv2.imshow("Calibrar OMR - Selecciona burbujas", clone)

    cv2.imshow("Calibrar OMR - Selecciona burbujas", clone)
    cv2.setMouseCallback("Calibrar OMR - Selecciona burbujas", mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rx0, ry0, rx1, ry1

# ---------------------------------------------------------
# 5) CARGAR CONFIG
# ---------------------------------------------------------
def cargar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return None

def guardar_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

# ---------------------------------------------------------
# 6) DETECCIÓN 20 PREGUNTAS (USANDO AREA CALIBRADA)
# ---------------------------------------------------------
def detectar_respuestas_20(warped, area):
    x0, y0, x1, y1 = area
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
            centro = celda[int(ch*0.3):int(ch*0.7), int(cw*0.3):int(cw*0.7)]
            negro = cv2.countNonZero(centro)
            valores.append(negro)

        max_val = max(valores)
        idx = valores.index(max_val)
        media = np.mean(valores)

        if max_val < media * 1.5:
            respuestas.append(None)
        else:
            respuestas.append(letras[idx])

    return respuestas

# ---------------------------------------------------------
# 7) MAIN
# ---------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Uso: omr_local.py <imagen> [--calibrar]"}))
        return

    ruta = sys.argv[1]
    modo_calibrar = "--calibrar" in sys.argv

    img = cv2.imread(ruta)
    if img is None:
        print(json.dumps({"ok": False, "error": "No se pudo leer la imagen"}))
        return

    codigo, id_examen, id_alumno, fecha_qr = leer_qr(img)

    corners = encontrar_cuadrados(img)
    warped = warp_hoja(img, corners) if corners is not None else img.copy()

    if modo_calibrar:
        area = seleccionar_area(warped)
        guardar_config({"area": area})
        print("Área guardada:", area)
        return

    cfg = cargar_config()
    if not cfg:
        print(json.dumps({"ok": False, "error": "Primero ejecuta en modo calibración"}))
        return

    area = cfg["area"]
    respuestas = detectar_respuestas_20(warped, area)

    _, buffer = cv2.imencode(".jpg", warped)
    debug_b64 = base64.b64encode(buffer).decode()

    print(json.dumps({
        "ok": True,
        "codigo": codigo,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha_qr": fecha_qr,
        "respuestas": respuestas,
        "warp_ok": True,
        "debug_image": debug_b64
    }))

if __name__ == "__main__":
    main()
