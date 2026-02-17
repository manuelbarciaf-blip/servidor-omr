import cv2
import numpy as np
import json
import sys
import base64

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

    # 3) Barcode lineal (detección simple)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
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
    x, y, w, h = cv2.boundingRect(c)

    roi = gray[y:y+h, x:x+w]

    # OCR simple para números
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
# DETECCIÓN DE BURBUJAS + IMAGEN DE DEPURACIÓN
# ============================================================

def detectar_respuestas(img):
    debug = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    burbujas = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 100 < area < 3000:
            x,y,w,h = cv2.boundingRect(c)
            ratio = w/float(h)
            if 0.6 < ratio < 1.4:
                burbujas.append((x,y,w,h))

    if not burbujas:
        return [], debug

    burbujas = sorted(burbujas, key=lambda b: (b[1], b[0]))

    respuestas = []
    opciones = ['A','B','C','D']

    filas = []
    fila_actual = [burbujas[0]]

    for b in burbujas[1:]:
        _, y_prev, _, h_prev = fila_actual[-1]
        x,y,w,h = b

        if abs(y - y_prev) < h_prev * 0.6:
            fila_actual.append(b)
        else:
            filas.append(sorted(fila_actual, key=lambda bb: bb[0]))
            fila_actual = [b]

    if fila_actual:
        filas.append(sorted(fila_actual, key=lambda bb: bb[0]))

    for fila in filas:
        if len(fila) != 4:
            continue

        intensidades = []
        for (x,y,w,h) in fila:
            roi = gray[y:y+h, x:x+w]
            mean = cv2.mean(roi)[0]
            intensidades.append(mean)

        idx = int(np.argmin(intensidades))
        respuestas.append(opciones[idx])

        # Dibujar burbujas en debug
        for i, (x,y,w,h) in enumerate(fila):
            color = (0,255,0) if i == idx else (0,0,255)
            cv2.rectangle(debug, (x,y), (x+w, y+h), color, 2)

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

    # Convertir imagen de depuración a base64
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

