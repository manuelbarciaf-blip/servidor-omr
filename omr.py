import cv2
import numpy as np
from pyzbar.pyzbar import decode
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import base64
import uvicorn

app = FastAPI(title="OMR API RRHHClases")

# =========================
# CONFIG PLANTILLA REAL
# =========================
QR_REGION = {
    "x0": 0.60,  # derecha
    "y0": 0.00,
    "x1": 1.00,
    "y1": 0.22   # zona superior donde está tu QR
}

# Burbujas en el CENTRO del A4 (tu nuevo diseño)
OMR_REGION = {
    "x0": 0.30,
    "y0": 0.22,  # empiezan en tercio superior (como dijiste)
    "x1": 0.70,
    "y1": 0.95
}

OPCIONES = ["A", "B", "C", "D"]
PREGUNTAS_POR_HOJA = 20

# Umbrales calibrados:
# - burbuja negra
# - boli azul/negro
# - foto móvil JPG
UMBRAL_MARCA = 0.32
UMBRAL_DOBLE = 0.75
UMBRAL_VACIO = 0.10
UMBRAL_DEBIL = 0.18


# =========================
# NORMALIZAR A4 (CLAVE)
# =========================
def normalizar_a4(img):
    return cv2.resize(img, (2480, 3508))


# =========================
# LECTURA QR ROBUSTA (2x2 cm)
# =========================
def leer_qr(img):
    # Intento 1: hoja completa
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    codes = decode(gray)
    if codes:
        return codes[0].data.decode("utf-8").strip()

    # Intento 2: zona superior derecha ampliada
    h, w = img.shape[:2]
    x0 = int(w * QR_REGION["x0"])
    y0 = int(h * QR_REGION["y0"])
    x1 = int(w * QR_REGION["x1"])
    y1 = int(h * QR_REGION["y1"])

    qr_crop = img[y0:y1, x0:x1]

    if qr_crop.size == 0:
        return None

    # Escalar para QR pequeño
    qr_crop = cv2.resize(qr_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray_qr = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
    gray_qr = cv2.GaussianBlur(gray_qr, (3, 3), 0)

    codes = decode(gray_qr)
    if not codes:
        return None

    return codes[0].data.decode("utf-8").strip()


# =========================
# DETECTAR TINTA AZUL/NEGRA (tipo Gravic)
# =========================
def preparar_omr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Azul (bolígrafo)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Negro (bolígrafo)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask = cv2.bitwise_or(mask_blue, mask_black)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, th = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    return th


# =========================
# RECORTE PORCENTAJE
# =========================
def recortar(img, region):
    h, w = img.shape[:2]
    return img[
        int(h * region["y0"]):int(h * region["y1"]),
        int(w * region["x0"]):int(w * region["x1"])
    ]


# =========================
# DETECTOR OMR GRAVIC (1 COLUMNA CENTRAL)
# =========================
def detectar_respuestas(zona_bin, zona_color, total=20):
    h, w = zona_bin.shape
    alto_fila = int(h / PREGUNTAS_POR_HOJA)
    ancho_op = int(w / 4)

    respuestas = []
    mapa = zona_color.copy()

    for i in range(min(total, PREGUNTAS_POR_HOJA)):
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
            total_pix = celda.size
            densidad = tinta / total_pix if total_pix > 0 else 0

            densidades.append(densidad)
            coords.append((x0, y0, x1, y1))

        max_d = max(densidades)
        idx = densidades.index(max_d)
        segundo = sorted(densidades, reverse=True)[1]

        # Vacía
        if max_d < UMBRAL_VACIO:
            respuestas.append("")
            continue

        # Doble marca (inválida)
        if segundo > max_d * UMBRAL_DOBLE:
            respuestas.append("X")
            for (x0, y0, x1, y1) in coords:
                cv2.rectangle(mapa, (x0, y0), (x1, y1), (0, 0, 255), 2)
            continue

        # Marca débil
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


# =========================
# ENDPOINT UNIVERSAL (PHP COMPATIBLE)
# =========================
@app.post("/corregir_omr")
async def corregir_omr(request: Request, file: UploadFile = File(None)):
    try:
        # CASO 1: PHP envía archivo (multipart)
        if file is not None:
            contenido = await file.read()
        else:
            # CASO 2: PHP envía binario crudo (muy común)
            contenido = await request.body()

        if not contenido:
            return JSONResponse({"ok": False, "error": "Imagen vacía"})

        # Decodificar imagen móvil JPG/PNG
        npimg = np.frombuffer(contenido, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"ok": False, "error": "Imagen no válida"})

        # 1. Normalizar a A4
        img_a4 = normalizar_a4(img)

        # 2. Leer QR (antes de OMR)
        qr = leer_qr(img_a4)
        if not qr:
            return JSONResponse({"ok": False, "error": "QR no detectado"})

        partes = qr.split("|")
        id_examen = partes[0] if len(partes) > 0 else None
        id_alumno = partes[1] if len(partes) > 1 else None
        fecha = partes[2] if len(partes) > 2 else None

        # 3. Preprocesado OMR
        th = preparar_omr(img_a4)

        # 4. Recorte zona central de burbujas
        zona_bin = recortar(th, OMR_REGION)
        zona_color = recortar(img_a4, OMR_REGION)

        # 5. Detectar respuestas (20 por hoja)
        respuestas, mapa = detectar_respuestas(zona_bin, zona_color, 20)

        # Debug visual (para tu PHP)
        _, buf = cv2.imencode(".jpg", mapa)
        debug_map = base64.b64encode(buf).decode()

        return {
            "ok": True,
            "qr": qr,
            "id_examen": id_examen,
            "id_alumno": id_alumno,
            "fecha": fecha,
            "respuestas": respuestas,
            "debug_map": debug_map
        }

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# =========================
# START DOCKER
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
