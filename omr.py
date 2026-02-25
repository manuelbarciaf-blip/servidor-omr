import cv2
import numpy as np
from pyzbar.pyzbar import decode
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import base64

app = FastAPI()

# ============================
# CONFIG PLANTILLA A4 (CENTRO + QR ARRIBA DERECHA)
# ============================

# Región QR (más grande para QR de 2x2 cm)
QR_REGION = {
    "x0": 0.60,
    "y0": 0.00,
    "x1": 1.00,
    "y1": 0.25
}

# Columna central de burbujas (como rediseñaste)
OMR_REGION = {
    "x0": 0.30,
    "y0": 0.20,   # tercio superior (como dijiste)
    "x1": 0.70,
    "y1": 0.95
}

OPCIONES = ["A", "B", "C", "D"]
PREGUNTAS_POR_HOJA = 20

# Umbrales estilo GRAVIC (burbujas negras + boli azul)
UMBRAL_MARCA = 0.35
UMBRAL_DOBLE = 0.75
UMBRAL_VACIO = 0.12
UMBRAL_DEBIL = 0.20


# ============================
# NORMALIZAR A4 (CLAVE OMR PROFESIONAL)
# ============================
def normalizar_a4(img):
    # Tamaño A4 estándar alta resolución
    return cv2.resize(img, (2480, 3508))


# ============================
# LECTURA QR ROBUSTA (NO FALLA CON QR PEQUEÑO)
# ============================
def leer_qr(img):
    # 1. Intento en imagen completa (más fiable)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray_full)

    if qr_codes:
        return qr_codes[0].data.decode("utf-8").strip()

    # 2. Intento en zona superior derecha ampliada
    h, w = img.shape[:2]
    x0 = int(w * QR_REGION["x0"])
    y0 = int(h * QR_REGION["y0"])
    x1 = int(w * QR_REGION["x1"])
    y1 = int(h * QR_REGION["y1"])

    qr_crop = img[y0:y1, x0:x1]

    # Escalar para mejorar detección de QR 2x2 cm
    qr_crop = cv2.resize(qr_crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    qr_codes = decode(gray)

    if not qr_codes:
        return None

    return qr_codes[0].data.decode("utf-8").strip()


# ============================
# PREPROCESADO OMR TIPO GRAVIC
# (detecta tinta azul/negra en burbuja negra)
# ============================
def preparar_omr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Azul (bolígrafo)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Negro (bolígrafo)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask = cv2.bitwise_or(mask_blue, mask_black)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, th = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    return th


# ============================
# RECORTE PORCENTAJE (PLANTILLA FIJA)
# ============================
def recortar_region(img, region):
    h, w = img.shape[:2]
    return img[
        int(h * region["y0"]):int(h * region["y1"]),
        int(w * region["x0"]):int(w * region["x1"])
    ]


# ============================
# DETECCIÓN GRAVIC (1 COLUMNA CENTRAL, 4 OPCIONES)
# ============================
def detectar_respuestas(zona_bin, zona_color, num_preguntas):
    h, w = zona_bin.shape
    alto_fila = int(h / PREGUNTAS_POR_HOJA)
    ancho_op = int(w / 4)

    respuestas = []
    mapa = zona_color.copy()

    for i in range(min(PREGUNTAS_POR_HOJA, num_preguntas)):
        y0 = i * alto_fila
        y1 = y0 + alto_fila

        fila = zona_bin[y0:y1, :]
        densidades = []
        coords = []

        for j in range(4):
            x0 = j * ancho_op
            x1 = (j + 1) * ancho_op
            celda = fila[:, x0:x1]

            tinta = cv2.countNonZero(celda)
            total = celda.size
            densidad = tinta / total if total > 0 else 0

            densidades.append(densidad)
            coords.append((x0, y0, x1, y1))

        max_d = max(densidades)
        idx = densidades.index(max_d)
        segundo = sorted(densidades, reverse=True)[1]

        # Vacía
        if max_d < UMBRAL_VACIO:
            respuestas.append("")
            continue

        # Doble marca
        if segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
            for (x0, y0, x1, y1) in coords:
                cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 0, 255), 2)
            continue

        # Débil
        if max_d < UMBRAL_MARCA:
            respuestas.append("?")
            x0, y0, x1, y1 = coords[idx]
            cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 255), 3)
            continue

        # Válida
        respuestas.append(OPCIONES[idx])
        x0, y0, x1, y1 = coords[idx]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa


# ============================
# ENDPOINT API PARA PHP
# ============================
@app.post("/corregir")
async def corregir(file: UploadFile = File(...)):
    contenido = await file.read()
    npimg = np.frombuffer(contenido, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"ok": False, "error": "Imagen inválida"})

    # 1. Normalizar a A4
    img_a4 = normalizar_a4(img)

    # 2. Leer QR (ANTES del OMR)
    qr = leer_qr(img_a4)
    if not qr:
        return JSONResponse({"ok": False, "error": "QR no detectado"})

    partes = qr.split("|")
    id_examen = partes[0] if len(partes) > 0 else None
    id_alumno = partes[1] if len(partes) > 1 else None
    fecha = partes[2] if len(partes) > 2 else None

    # 3. Preprocesado OMR
    th = preparar_omr(img_a4)

    # 4. Recorte zona burbujas (CENTRO A4)
    zona_bin = recortar_region(th, OMR_REGION)
    zona_color = recortar_region(img_a4, OMR_REGION)

    # 5. Detectar respuestas (20 por hoja)
    respuestas, mapa = detectar_respuestas(zona_bin, zona_color, 20)

    # Debug visual
    _, buffer = cv2.imencode(".jpg", mapa)
    debug_map = base64.b64encode(buffer).decode()

    return {
        "ok": True,
        "qr": qr,
        "id_examen": id_examen,
        "id_alumno": id_alumno,
        "fecha": fecha,
        "respuestas": respuestas,
        "debug_map": debug_map
    }


# ============================
# ARRANQUE SERVIDOR (DOCKER)
# ============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
