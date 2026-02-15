from flask import Flask, request, jsonify
from PIL import Image
from pyzbar.pyzbar import decode
import numpy as np
import os
import json

app = Flask(__name__)

# Cargar plantilla layout
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")

    # ------------------------------
    # 1. Leer código de barras
    # ------------------------------
    codigos = decode(img)
    if codigos:
        codigo = codigos[0].data.decode("utf-8")
    else:
        codigo = "DESCONOCIDO"

    # ------------------------------
    # 2. Leer burbujas OMR
    # ------------------------------
    respuestas = {}
    for pregunta in layout["preguntas"]:
        num = pregunta["numero"]
        opciones = pregunta["opciones"]

        mejor_opcion = "BLANCO"
        mejor_nivel = 0

        for op in opciones:
            x1, y1, x2, y2 = op["bbox"]
            recorte = img.crop((x1, y1, x2, y2))
            arr = np.array(recorte.convert("L"))
            nivel = 255 - arr.mean()

            if nivel > mejor_nivel and nivel > layout["umbral"]:
                mejor_nivel = nivel
                mejor_opcion = op["letra"]

        respuestas[num] = mejor_opcion

    return jsonify({
        "codigo": codigo,
        "respuestas": respuestas,
        "num_preguntas": len(respuestas)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
