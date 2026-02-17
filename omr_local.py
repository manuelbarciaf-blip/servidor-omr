import cv2
import numpy as np
import json
import sys
import base64

# ============================================================
# PARÁMETROS DEL DISEÑO (A4 → 1700x2338 px)
# ============================================================

IMG_W = 1700
IMG_H = 2338

# Escalas aproximadas (px/mm)
SX = IMG_W / 210.0   # ≈ 8.095 px/mm
SY = IMG_H / 297.0   # ≈ 7.87 px/mm

# Cuadrados de referencia (en mm desde bordes)
# TL: 15 mm izq, 22 mm sup
# TR: 20 mm dcha, 22 mm sup
# BL: 15 mm izq, 22 mm inf
# BR: 20 mm dcha, 22 mm inf

def mm_to_px_x(mm):
    return mm * SX

def mm_to_px_y(mm):
    return mm * SY

def get_corner_points():
    # Coordenadas aproximadas en píxeles
    tl = (int(mm_to_px_x(15)), int(mm_to_px_y(22)))
    tr = (IMG_W - int(mm_to_px_x(20)), int(mm_to_px_y(22)))
    bl = (int(mm_to_px_x(15)), IMG_H - int(mm_to_px_y(22)))
    br = (IMG_W - int(mm_to_px_x(20)), IMG_H - int(mm_to_px_y(22)))
    return tl, tr, bl, br


# ============================================================
# LECTOR DE BARCODE (QR + BARCODE LINEAL)
# ============================================================

def leer_barcode(img):
    detector = cv2.QRCodeDetector()

    # 1) QR normal
    data, points, _ = detector.detectAndDecode(img)
    if data:
        return data.strip()

    # 2) QR invertido
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    data2, _, _ = detector.detectAndDecode(inv)
    if data2:
        return data2.strip()

    # 3) Barcode lineal (detección simple en franja superior)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi_bar = gray[0:int(h*0.25), :]  # franja superior

    gradX = cv2.Sobel(roi_bar, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradX = cv2.convertScaleAbs(gradX)

    _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    x, y, w2, h2 = cv2.boundingRect(c)

    roi = roi_bar[y:y+h2, x:x+w2]

    try:
        import pytesseract
        txt = pytesseract.image_to_string(roi, config="--psm 6 digits")
        txt = txt.strip()
        if txt:
            return txt
    except:
        pass

    return None


# ============================================================
# WARP PERSPECTIVE + RECORTE ZONA RESPUESTAS
# ============================================================

def normalizar_hoja(img):
    h, w = img.shape[:2]

    # Puntos fuente aproximados (en la imagen real)
    tl, tr, bl, br = get_corner_points()

    src = np.float32([
        [tl[0], tl[1]],
        [tr[0], tr[1]],
        [bl[0], bl[1]],
        [br[0], br[1]]
    ])

    dst = np.float32([
        [0, 0],
        [IMG_W - 1, 0],
        [0, IMG_H - 1],
        [IMG_W - 1, IMG_H - 1]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (IMG_W, IMG_H))

    return warped


def recortar_zona_respuestas(img_norm):
    # Estos valores se ajustan a tu diseño (aprox)
    # Puedes afinarlos si ves que la cuadrícula cae un poco desplazada
    x1 = int(IMG_W * 0.15)   # ~ 15% desde la izquierda
    x2 = int(IMG_W * 0.90)   # ~ 90% hacia la derecha
    y1 = int(IMG_H * 0.20)   # ~ 20% desde arriba
    y2 = int(IMG_H * 0.93)   # ~ 93% hacia abajo

    roi = img_norm[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


# ============================================================
# DETECCIÓN DE BURBUJAS POR CUADRÍCULA (HASTA 60 PREGUNTAS)
# ============================================================

def detectar_respuestas(img):
    # 1) Normalizar hoja
    norm = normalizar_hoja(img)
    debug = norm.copy()

    # 2) Recortar zona de respuestas
    roi, (rx1, ry1, rx2, ry2) = recortar_zona_respuestas(norm)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.equalizeHist(gray)

    # Máximo de filas que soporta el diseño
    MAX_FILAS = 60
    NUM_COLS = 4

    h, w = gray.shape
    fila_h = h / float(MAX_FILAS)
    col_w = w / float(NUM_COLS)

    respuestas = []
    opciones = ['A', 'B', 'C', 'D']

    for fila_idx in range(MAX_FILAS):
        y_start = int(fila_idx * fila_h)
        y_end   = int((fila_idx + 1) * fila_h)

        scores = []
        celdas = []

        for col_idx in range(NUM_COLS):
            x_start = int(col_idx * col_w)
            x_end   = int((col_idx + 1) * col_w)

            roi_bur = gray[y_start:y_end, x_start:x_end]
            if roi_bur.size == 0:
                scores.append(-1e9)
                celdas.append((x_start, y_start, x_end, y_end))
                continue

            mean = cv2.mean(roi_bur)[0]
            var = float(np.var(roi_bur))

            score = (255 - mean) + (var * 0.02)
            scores.append(score)
            celdas.append((x_start, y_start, x_end, y_end))

        # Si todas las puntuaciones son muy bajas, consideramos que no hay respuesta
        max_score = max(scores)
        if max_score < 5:  # umbral muy bajo, ajustable
            continue

        idx = int(np.argmax(scores))
        respuestas.append(opciones[idx])

        # Dibujar en debug (sobre la hoja normalizada)
        for i, (xs, ys, xe, ye) in enumerate(celdas):
            color = (0, 255, 0) if i == idx else (0, 0, 255)
            # trasladar coords a la imagen normalizada
            X1 = rx1 + xs
            Y1 = ry1 + ys
            X2 = rx1 + xe
            Y2 = ry1 + ye
            cv2.rectangle(debug, (X1, Y1), (X2, Y2), color, 2)

    return respuestas, debug


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Falta ruta de imagen"}))
        return

    ruta = sys.argv[1]
    img = cv2.imread(ruta)

    if img is None:
        print(json.dumps({"ok": False, "error": "Imagen inválida"}))
        return

    barcode = leer_barcode(img)
    respuestas, debug_img = detectar_respuestas(img)

    _, buffer = cv2.imencode(".jpg", debug_img)
    debug_b64 = base64.b64encode(buffer).decode("utf-8")

    print(json.dumps({
        "ok": True,
        "barcode": barcode,
        "respuestas": respuestas,
        "num_preguntas": len(respuestas),
        "debug_image": debug_b64
    }))


if __name__ == "__main__":
    main()
