from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)

# Cargar plantilla de layout
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

@app.route("/")
def home():
    return "Servidor OMR activo ✅"

@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file enviado"}), 400

        file = request.files["file"]

        # Abrir imagen en escala de grises
        img = Image.open(file.stream).convert("L")

        # ⚠️ TAMAÑO REALISTA (ANTES TENÍAS 1401400 Y CRASHEABA)
        ANCHO = 1000
        ALTO = 1400
        img = img.resize((ANCHO, ALTO))

        img_np = np.array(img)

        # Umbral de detección (ajustable)
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

                if celda.size == 0:
                    intensidades.append(0)
                    continue

                negro = np.mean(celda) * 100
                intensidades.append(negro)

            marcadas = sum(v > 25 for v in intensidades)

            if marcadas == 0:
                respuestas[str(i+1)] = "BLANCO"
            elif marcadas > 1:
                respuestas[str(i+1)] = "DOBLE"
            else:
                idx = int(np.argmax(intensidades))
                respuestas[str(i+1)] = opciones[idx]

        return jsonify({
            "ok": True,
            "respuestas": respuestas
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    # ⚠️ PUERTO DINÁMICO OBLIGATORIO EN RENDER
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
