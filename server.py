import cv2
import numpy as np
import json
import os
from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode

app = Flask(__name__)

# ============================
# CARGAR PLANTILLA
# ============================
with open("plantilla.json", "r") as f:
    PLANTILLA = json.load(f)

NUM_PREG = PLANTILLA["num_preguntas"]
OPCIONES = PLANTILLA["opciones"]
LAYOUT = PLANTILLA["layout"]

# ============================
# LECTOR DE BARCODE
# ============================
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

# ============================
# OMR POR PLANTILLA
# ============================
def detectar_respuestas_por_plantilla(img):

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    zona = LAYOUT["zona_respuestas"]

    x0 = int(zona["x0_rel"] * w)
    y0 = int(zona["y0_rel"] * h)
    ancho = int(zona["ancho_rel"] * w)
    alto = int(zona["alto_rel"] * h)

    zona_img = gray[y0:y0+alto, x0:x0+ancho]

    respuestas = []

    for i in range(NUM_PREG):

        fila_info = LAYOUT["filas"]
        y_fila = int(fila_info["offset_y_rel"] * alto + i * fila_info["alto_celda_rel"] * alto)

        intensidades = []

        for opcion in OPCIONES:
            col_info = LAYOUT["columnas"][opcion]

            x_col = int(col_info["offset_x_rel"] * ancho)

            cas = LAYOUT["casilla"]
            cw = int(cas["ancho_rel"] * ancho)
            ch = int(cas["alto_rel"] * alto)

            roi = zona_img[y_fila:y_fila+ch, x_col:x_col+cw]

            mean = cv2.mean(roi)[0]
            intensidades.append(mean)

        idx = int(np.argmin(intensidades))
        respuestas.append(OPCIONES[idx])

    return respuestas

# ============================
# ENDPOINT API
# ============================
@app.route("/omr/leer", methods=["POST"])
def omr_leer():

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No se envió archivo"})

    file = request.files["file"]
    ruta_temp = "temp.jpg"
    file.save(ruta_temp)

    img = cv2.imread(ruta_temp)

    if img is None:
        return jsonify({"ok": False, "error": "Imagen inválida"})

    barcode = leer_barcode(img)
    respuestas = detectar_respuestas_por_plantilla(img)

    os.remove(ruta_temp)

    return jsonify({
        "ok": True,
        "barcode": barcode,
        "respuestas": respuestas,
        "num_preguntas": len(respuestas)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
