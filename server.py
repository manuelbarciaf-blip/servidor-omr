from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import json
import io

app = Flask(__name__)

# Cargar layout
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    file = request.files["file"]

    try:
        # Abrir imagen y corregir orientación (clave para móviles)
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")  # escala de grises
    except:
        return jsonify({"ok": False, "error": "Imagen inválida"}), 400

    # 🔥 TAMAÑO REALISTA (A4 escaneado / móvil)
    ANCHO = 1000
    ALTO = 1400  # <-- CORREGIDO (antes tenías 1401400)

    img = img.resize((ANCHO, ALTO))
    img_np = np.array(img)

    # Mejor umbral para fotos móviles (más tolerante)
    thresh = img_np < 160  # antes 180 (demasiado agresivo)

    lay = layout["layout"]

    # Zona de respuestas (relativa)
    x0 = int(lay["zona_respuestas"]["x0_rel"] * ANCHO)
    y0 = int(lay["zona_respuestas"]["y0_rel"] * ALTO)
    ancho = int(lay["zona_respuestas"]["ancho_rel"] * ANCHO)
    alto = int(lay["zona_respuestas"]["alto_rel"] * ALTO)

    # Protección contra recortes fuera de rango (fotos torcidas)
    x1 = min(x0 + ancho, ANCHO)
    y1 = min(y0 + alto, ALTO)

    zona = thresh[y0:y1, x0:x1]

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

        # Evitar overflow si la foto está desplazada
        if fila_y + 5 >= zona.shape[0]:
            respuestas[str(i+1)] = "BLANCO"
            continue

        intensidades = []

        for letra in opciones:
            col_x = int(col_offsets[letra] * ancho)
            w = int(ancho_celda * ancho)
            h = int(alto_celda * alto)

            # Protección de límites (crucial en fotos móviles)
            x_end = min(col_x + w, zona.shape[1])
            y_end = min(fila_y + h, zona.shape[0])

            celda = zona[fila_y:y_end, col_x:x_end]

            if celda.size == 0:
                intensidades.append(0)
                continue

            # % de píxeles negros (mejor que media)
            negro = (np.sum(celda) / celda.size) * 100
            intensidades.append(negro)

        marcadas = sum(v > 20 for v in intensidades)  # umbral más realista

        if marcadas == 0:
            respuestas[str(i+1)] = "BLANCO"
        elif marcadas > 1:
            respuestas[str(i+1)] = "DOBLE"
        else:
            idx = np.argmax(intensidades)
            respuestas[str(i+1)] = opciones[idx]

    return jsonify({
        "ok": True,
        "respuestas": respuestas
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
