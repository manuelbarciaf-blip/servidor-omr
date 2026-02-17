import cv2
import numpy as np
import json
import sys
import base64

# Si usas pyzbar para código de barras:
try:
    from pyzbar.pyzbar import decode as decode_barcode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

IMG_W = 1700
IMG_H = 2338

# ---------------------------------------------------------
# LECTURA CÓDIGO DE BARRAS (LINEAL) + QR (POR SI LO USAS)
# ---------------------------------------------------------

def leer_codigo(img):
    # 1) Intentar con pyzbar (código de barras + QR)
    if HAS_PYZBAR:
        decoded = decode_barcode(img)
        if decoded:
            # Tomamos el primero
            data = decoded[0].data.decode("utf-8").strip()
            return data

    # 2) Intentar QRCodeDetector como respaldo
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if data:
        return data.strip()

    # 3) Intentar QR invertido
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    data2, _, _ = detector.detectAndDecode(inv)
    if data2:
        return data2.strip()

    return None

# ---------------------------------------------------------
# DETECCIÓN DE LOS 4 CUADRADOS DE ESQUINA
# ---------------------------------------------------------

def encontrar_cuadrados(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertimos si los cuadrados son negros
    th_inv = 255 - th

    contours, _ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    h, w = gray.shape

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500 or area > 20000:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            ratio = cw / float(ch)
            if 0.7 < ratio < 1.3:
                candidatos.append((x, y, cw, ch, approx))

    if len(candidatos) < 4:
        return None

    # Ordenar por posición (TL, TR, BL, BR)
    centros = []
    for (x, y, cw, ch, approx) in candidatos:
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        centros.append((cx, cy, approx))

    # TL: min x+min y, TR: max x+min y, BL: min x+max y, BR: max x+max y
    centros_sorted = sorted(centros, key=lambda p: (p[1], p[0]))  # primero por y, luego x
    top = centros_sorted[:2]
    bottom = centros_sorted[2:]

    tl = min(top, key=lambda p: p[0])
    tr = max(top, key=lambda p: p[0])
    bl = min(bottom, key=lambda p: p[0])
    br = max(bottom, key=lambda p: p[0])

    def centro_aprox(ap):
        ap = ap.reshape(-1, 2)
        return np.mean(ap, axis=0)

    TL = centro_aprox(tl[2])
    TR = centro_aprox(tr[2])
    BL = centro_aprox(bl[2])
    BR = centro_aprox(br[2])

    return np.float32([TL, TR, BL, BR])

# ---------------------------------------------------------
# WARP PERSPECTIVE A PLANTILLA NORMALIZADA
# ---------------------------------------------------------

def normalizar_hoja(img):
    pts = encontrar_cuadrados(img)
    if pts is None:
        # Si no encontramos cuadrados, devolvemos la imagen tal cual
        return img, False

    dst = np.float32([
        [0, 0],
        [IMG_W - 1, 0],
        [0, IMG_H - 1],
        [IMG_W - 1, IMG_H - 1]
    ])

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (IMG_W, IMG_H))

    return warped, True

# ---------------------------------------------------------
# RECORTE ZONA RESPUESTAS (AJUSTABLE)
# ---------------------------------------------------------

def recortar_zona_respuestas(img_norm):
    # Estos valores se ajustan una vez veas el debug
    # y deben dejar dentro TODAS las burbujas
    h, w = img_norm.shape[:2]

    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    y1 = int(h * 0.22)
    y2 = int(h * 0.90)

    roi = img_norm[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

# ---------------------------------------------------------
# DETECCIÓN DE BURBUJAS REDONDAS POR CUADRÍCULA
# ---------------------------------------------------------

def detectar_respuestas(img, max_preguntas=60):
    norm, warped_ok = normalizar_hoja(img)
    debug = norm.copy()

    roi, (rx1, ry1, rx2, ry2) = recortar_zona_respuestas(norm)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.equalizeHist(gray)

    h, w = gray.shape
    NUM_COLS = 4
    MAX_FILAS = max_preguntas

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

        max_score = max(scores)
        if max_score < 5:
            continue

        idx = int(np.argmax(scores))
        respuestas.append(opciones[idx])

        for i, (xs, ys, xe, ye) in enumerate(celdas):
            color = (0, 255, 0) if i == idx else (0, 0, 255)
            X1 = rx1 + xs
            Y1 = ry1 + ys
            X2 = rx1 + xe
            Y2 = ry1 + ye
            cv2.rectangle(debug, (X1, Y1), (X2, Y2), color, 1)

    return respuestas, debug, warped_ok

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Falta ruta de imagen"}))
        return

    ruta = sys.argv[1]
    img = cv2.imread(ruta)

    if img is None:
        print(json.dumps({"ok": False, "error": "Imagen inválida"}))
        return

    codigo = leer_codigo(img)
    respuestas, debug_img, warped_ok = detectar_respuestas(img)

    _, buffer = cv2.imencode(".jpg", debug_img)
    debug_b64 = base64.b64encode(buffer).decode("utf-8")

    print(json.dumps({
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "num_preguntas_detectadas": len(respuestas),
        "warp_ok": warped_ok,
        "debug_image": debug_b64
    }))


if __name__ == "__main__":
    main()
