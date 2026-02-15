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
from flask import send_file
from PIL import ImageDraw

@app.route("/omr/ver_plantilla")
def ver_plantilla():
    # Cargar plantilla
    with open("plantilla_layout.json", "r") as f:
        plantilla = json.load(f)

    # Crear imagen base (blanca)
    img = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(img)

    # Colores por opción
    colores = {
        "A": "red",
        "B": "blue",
        "C": "green",
        "D": "orange"
    }

    # Dibujar cada pregunta
    for pregunta in plantilla["preguntas"]:
        num = pregunta["numero"]

        for op in pregunta["opciones"]:
            letra = op["letra"]
            x1, y1, x2, y2 = op["bbox"]

            # Dibujar rectángulo
            draw.rectangle([x1, y1, x2, y2], outline=colores[letra], width=3)

            # Etiqueta
            draw.text((x1, y1 - 12), f"{num}{letra}", fill=colores[letra])

    # Guardar temporalmente
    ruta = "plantilla_preview.png"
    img.save(ruta)

    return send_file(ruta, mimetype="image/png")
