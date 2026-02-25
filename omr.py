import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import base64
import uvicorn

app = FastAPI()

# =========================================
# CONFIG REAL DE TU HOJA (NUEVO DISEÑO)
# =========================================
QR_REGION = {
    "x0": 0.00,   # QR a la izquierda
    "y0": 0.00,
    "x1": 0.35,
    "y1": 0.25
}

OMR_REGION = {
    "x0": 0.30,
    "y0": 0.22,
    "x1": 0.70,
    "y1": 0.95
}

OPCIONES = ["A", "B", "C", "D"]
PREGUNTAS_POR_HOJA = 20

# =========================================
# UMBRALES AJUSTADOS
# =========================================
UMBRAL_MARCA = 0.30
UMBRAL_DOBLE = 0.75
UMBRAL_VACIO = 0.05   # detecta marcas suaves


# =========================================
# NORMALIZAR A4
# =========================================
def normalizar_a4(img):
    return cv2.resize(img, (2480, 3508))


# =========================================
# LECTURA QR
# =========================================
def leer_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    codes = zbar_decode(gray)
    if codes:
        return codes[0].data.decode("utf-8").strip()

    h, w = img.shape[:2]
    x0 = int(w * QR_REGION["x0"])
    y0 = int(h * QR_REGION["y0"])
    x1 = int(w * QR_REGION["x1"])
    y1 = int(h * QR_REGION["y1"])

    qr_crop = img[y0:y1, x0:x1]
    if qr_crop.size == 0:
        return None

    qr_crop = cv2.resize(qr_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray_qr = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
    gray_qr = cv2.GaussianBlur(gray_qr, (3, 3), 0)

    codes = zbar_decode(gray_qr)
    if not codes:
        return None

    return codes[0].data.decode("utf-8").strip()


# =========================================
# MÁSCARA DE TINTA (MEJORADA)
# =========================================
def preparar_mascara_tinta(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Azul claro + azul normal
    lower_blue = np.array([70, 10, 10])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Negro de bolígrafo (evitando burbuja impresa)
    lower_black = np.array([0, 0, 40])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask = cv2.bitwise_or(mask_blue, mask_black)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, th = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    return th


# =========================================
# RECORTE POR PORCENTAJE
# =========================================
def recortar(img, region):
    h, w = img.shape[:2]
    return img[
        int(h * region["y0"]):int(h * region["y1"]),
        int(w * region["x0"]):int(w * region["x1"])
    ]


# =========================================
# DETECCIÓN OMR
# =========================================
def detectar_respuestas(zona_bin, zona_color, total_preguntas=20):
    h, w = zona_bin.shape
    alto_fila = int(h / PREGUNTAS_POR_HOJA)
    ancho_op = int(w / 4)

    respuestas = []
    mapa = zona_color.copy()

    for i in range(min(total_preguntas, PREGUNTAS_POR_HOJA)):
        y0 = i * alto_fila
        y1 = y0 + alto_fila

        fila = zona_bin[y0:y1, :]
        if fila.size == 0:
            respuestas.append("")
            continue

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

        if max_d < UMBRAL_VACIO:
            respuestas.append("")
            continue

        if segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
            for (x0, y0, x1, y1) in coords:
                cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 0, 255), 2)
            continue

        respuestas.append(OPCIONES[idx])
        x0, y0, x1, y1 = coords[idx]
        cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return respuestas, mapa


# =========================================
# ENDPOINT PRINCIPAL
# =========================================
@app.post("/")
async def procesar_omr(request: Request):
    try:
        binario = await request.body()

        # DEBUG OPCIONAL: ver si la imagen llega bien
        # with open("debug_recibido.jpg", "wb") as f:
        #     f.write(binario)

        if not binario:
            return {"ok": False, "error": "Imagen vacía"}

        npimg = np.frombuffer(binario, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"ok": False, "error": "No se pudo decodificar la imagen"}

        img_a4 = normalizar_a4(img)

        codigo = leer_qr(img_a4)
        if not codigo:
            return {"ok": False, "error": "QR no detectado"}

        th = preparar_mascara_tinta(img_a4)

        zona_bin = recortar(th, OMR_REGION)
        zona_color = recortar(img_a4, OMR_REGION)

        respuestas, mapa = detectar_respuestas(zona_bin, zona_color, 20)

        _, buffer = cv2.imencode(".jpg", zona_color)
        debug_image = base64.b64encode(buffer).decode()

        return {
            "ok": True,
            "codigo": codigo,
            "respuestas": respuestas,
            "debug_image": debug_image
        }

    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
