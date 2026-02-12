from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import json

app = Flask(__name__)

with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("L")

    ANCHO = 1000
    ALTO = 1401400
    img = img.resize((ANCHO, ALTO))
    img_np = np.array(img)

    thresh = img_np < 180

    lay = layout["layout"]

    x0 = int(lay["zona_respuestas"]["x0_rel"] * ANCHO)
    y0 = int(lay["zona_respuestas"]["y0_rel"] * ALTO)
    ancho = int(lay["zona_respuestas"]["ancho_rel"] * ANCHO)
    alto = int(lay["zona_respuestas"]["alto_rel"] * ALTO)

    zona = thresh[y0:y0+alto, x0:x0+ancho]

    offset_y = lay["filas"]["offset_y_rel"]
    alto_celda = lay["filas"]["alto_celda_rel"]
    ancho_celda = lay["casilla"]["ancho_rel"]

    col_offsets = {
        "A": lay["columnas"]["A"]["offset_x_rel"],
        "B": lay["columnas"]["B"]["offset_x_rel"],
        "C": lay["columnas"]["C"]["offset_x_rel"],
        "D": lay["columnas"]["D"]["offset_x_rel"]
    }

    num_preguntas = layout["num_preguntas"]
    opciones = layout["opciones"]

    respuestas = {}

    for i in range(num_preguntas):
        fila_y = int(i * offset_y * alto)

        intensidades = []

        for letra in opciones:
            col_x = int(col_offsets[letra] * ancho)
            w = int(ancho_celda * ancho)
            h = int(alto_celda * alto)

            celda = zona[fila_y:fila_y+h, col_x:col_x+w]
            negro = np.mean(celda) * 100
            intensidades.append(negro)

        marcadas = sum(v > 25 for v in intensidades)

        if marcadas == 0:
            respuestas[str(i+1)] = "BLANCO"
        elif marcadas > 1:
            respuestas[str(i+1)] = "DOBLE"
        else:
            idx = np.argmax(intensidades)
            respuestas[str(i+1)] = opciones[idx]

    return jsonify({"ok": True, "respuestas": respuestas})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
