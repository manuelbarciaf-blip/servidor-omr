import cv2
import numpy as np
import json
import sys
from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode
import os

app = Flask(__name__)

# =========================
# LECTOR DE BARCODE
# =========================
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

# =========================
# PARSEAR CODIGO
# =========================
def parsear_codigo(codigo):
    try:
        return {
            "id_examen": codigo[0:7],
            "id_alumno": codigo[7:13],
            "fecha": codigo[13:]
        }
    except:
        return {
            "id_examen": None,
            "id_alumno": None,
            "fecha": None
        }

# =========================
# DETECTAR RESPUESTAS
# =========================
def detectar_respuestas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    burbujas = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 150 < area < 2000:
            x,y,w,h = cv2.boundingRect(c)
            ratio = w/float(h)
            if 0.7 < ratio < 1.3:
                burbujas.append((x,y,w,h))

    burbujas = sorted(burbujas, key=lambda b: (b[1], b[0]))

    respuestas = []
    opciones = ['A','B','C','D']

    fila = []
    last_y = None

    for b in burbujas:
        x,y,w,h = b

        if last_y is None:
            last_y = y

        if abs(y - last_y) > 20:
            if len(fila) == 4:
                respuestas.append(evaluar_fila(img, fila, opciones))
            fila = []
            last_y = y

        fila.append(b)

        if len(fila) == 4:
            respuestas.append(evaluar_fila(img, fila, opciones))
            fila = []

    return respuestas

def evaluar_fila(img, fila, opciones):
    intensidades = []
    for (x,y,w,h) in fila:
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean = cv2.mean(gray)[0]
        intensidades.append(mean)

    idx = np.argmin(intensidades)
    return opciones[idx]

# =========================
# DIBUJAR CORRECCIÓN
# =========================
def dibujar_correccion(img, respuestas):
    salida = img.copy()
    y = 50
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, r in enumerate(respuestas):
        cv2.putText(salida, f"{i+1}:{r}", (50, y), font, 0.6, (0,0,255), 2)
        y += 25
    
    return salida

# =========================
# ENDPOINT API
# =========================
@app.route("/omr/leer", methods=["POST"])
def omr_leer():
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No se envió archivo"})

    file = request.files['file']
    ruta_temp = "temp.jpg"
    file.save(ruta_temp)

    img = cv2.imread(ruta_temp)

    if img is None:
        return jsonify({"ok": False, "error": "Imagen inválida"})

    codigo = leer_barcode(img)
    datos = parsear_codigo(codigo) if codigo else {"id_examen": None, "id_alumno": None, "fecha": None}

    respuestas = detectar_respuestas(img)

    corregida = dibujar_correccion(img, respuestas)
    ruta_corregida = "corregido.jpg"
    cv2.imwrite(ruta_corregida, corregida)

    os.remove(ruta_temp)

    return jsonify({
        "ok": True,
        "barcode": codigo,
        "datos": datos,
        "num_preguntas_detectadas": len(respuestas),
        "respuestas": respuestas,
        "imagen_corregida": ruta_corregida
    })

# =========================
# RUN SERVER (RENDER)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
