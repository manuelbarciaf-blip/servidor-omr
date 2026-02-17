import cv2
import numpy as np
import json
import sys
from pyzbar.pyzbar import decode

def leer_barcode(img):
    try:
        barcodes = decode(img)
        for barcode in barcodes:
            codigo = barcode.data.decode("utf-8")
            if codigo:
                return codigo
        return None
    except:
        return None

def detectar_respuestas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Normalizar iluminación
    gray = cv2.equalizeHist(gray)

    # 2) Binarización adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # 3) Limpiar ruido
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

    return respuestas

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
    respuestas = detectar_respuestas(img)

    print(json.dumps({
        "ok": True,
        "barcode": barcode,
        "respuestas": respuestas,
        "num_preguntas": len(respuestas)
    }))

if __name__ == "__main__":
    main()
