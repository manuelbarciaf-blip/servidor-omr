import cv2
import numpy as np
from pyzbar.pyzbar import decode

# -------------------------------
# CONFIGURACIÓN OMR AVANZADA
# -------------------------------
UMBRAL_MARCADA = 0.22      # densidad mínima para considerar marcada
UMBRAL_DEBIL = 0.12        # marca débil (revisar)
UMBRAL_DOBLE = 0.35        # si dos superan, doble marca
DEBUG = True               # guarda imagen debug

def detectar_qr(imagen):
    qr = decode(imagen)
    if qr:
        return qr[0].data.decode("utf-8")
    return None


def ordenar_puntos(pts):
    pts = pts.reshape(4, 2)
    suma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(suma)]
    rect[2] = pts[np.argmax(suma)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def deskew_hoja(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for c in contornos:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            rect = ordenar_puntos(approx)
            (tl, tr, br, bl) = rect

            ancho = 2100
            alto = 2970

            dst = np.array([
                [0, 0],
                [ancho, 0],
                [ancho, alto],
                [0, alto]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(imagen, M, (ancho, alto))
            return warp

    return imagen  # fallback


def obtener_zona_omr(hoja):
    h, w = hoja.shape[:2]

    # Zona izquierda donde están las burbujas (según tu plantilla)
    x1 = int(w * 0.05)
    x2 = int(w * 0.35)
    y1 = int(h * 0.15)
    y2 = int(h * 0.95)

    return hoja[y1:y2, x1:x2]


def calcular_densidad(burbuja):
    # Convertir a HSV para ignorar el rojo del borde
    hsv = cv2.cvtColor(burbuja, cv2.COLOR_BGR2HSV)

    # Detectar tinta azul y negra (no el rojo)
    mask_azul = cv2.inRange(hsv, (90, 50, 50), (140, 255, 255))
    gray = cv2.cvtColor(burbuja, cv2.COLOR_BGR2GRAY)
    _, mask_negro = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.bitwise_or(mask_azul, mask_negro)

    area = mask.size
    relleno = cv2.countNonZero(mask)

    return relleno / float(area)


def procesar_burbujas(zona, num_preguntas=20, opciones=4):
    h, w = zona.shape[:2]

    paso_y = h / num_preguntas
    paso_x = w / opciones

    respuestas = []
    debug_img = zona.copy()

    for i in range(num_preguntas):
        densidades = []

        for j in range(opciones):
            x1 = int(j * paso_x + paso_x * 0.2)
            x2 = int((j + 1) * paso_x - paso_x * 0.2)
            y1 = int(i * paso_y + paso_y * 0.2)
            y2 = int((i + 1) * paso_y - paso_y * 0.2)

            burbuja = zona[y1:y2, x1:x2]
            densidad = calcular_densidad(burbuja)
            densidades.append(densidad)

            if DEBUG:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Clasificación avanzada
        marcadas = [d for d in densidades if d > UMBRAL_MARCADA]

        if len(marcadas) == 0:
            respuestas.append("")  # en blanco
        elif len(marcadas) > 1:
            respuestas.append("X")  # doble marca (inválida)
        else:
            idx = np.argmax(densidades)
            respuestas.append(["A", "B", "C", "D"][idx])

    return respuestas, debug_img


def procesar_omr(binario):
    try:
        nparr = np.frombuffer(binario, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if imagen is None:
            return {"ok": False, "error": "Imagen no válida"}

        # 1. Detectar QR
        qr = detectar_qr(imagen)

        # 2. Enderezar hoja (deskew automático)
        hoja = deskew_hoja(imagen)

        # 3. Recortar zona OMR
        zona_omr = obtener_zona_omr(hoja)

        # 4. Detectar respuestas (densidad tipo Gravic)
        respuestas, debug = procesar_burbujas(zona_omr, num_preguntas=20)

        # Guardar debug para ver qué está leyendo
        if DEBUG:
            cv2.imwrite("debug_omr.jpg", debug)

        return {
            "ok": True,
            "qr": qr,
            "respuestas": respuestas
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
