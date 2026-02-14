# reconstrucción
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import json
import io

# 🔥 NUEVO: lector de código de barras
from pyzbar.pyzbar import decode

app = Flask(__name__)

# Cargar layout
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

def leer_codigo_barras(img):
    """
    Recorta la zona superior derecha donde está el código
    y usa pyzbar para leer CODE128.
    """

    W, H = img.size

    # Zona del código (ajustada a tu plantilla)
    crop_w = int(W * 0.40)
    crop_h = int(H * 0.15)
    crop_x = W - crop_w - 20
    crop_y = 20

    zona = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    # Intentar leer código
    barcodes = decode(zona)

    if not barcodes:
        return None

    # Tomar el primero
    return barcodes[0].data.decode("utf-8")


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
    ALTO = 1400

    img = img.resize((ANCHO, ALTO))
    img_np = np.array(img)

    # 🔥 LEER CÓDIGO DE BARRAS
    codigo = leer_codigo_barras(img)

    # Mejor umbral para fotos móviles
    thresh = img_np < 160

    lay = layout["layout"]

    # Zona de respuestas
    x0 = int(lay["zona_respuestas"]["x0_rel"] * ANCHO)
    y0 = int(lay["zona_respuestas"]["y0_rel"] * ALTO)
    ancho = int(lay["zona_respuestas"]["ancho_rel"] * ANCHO)
    alto = int(lay["zona_respuestas"]["alto_rel"] * ALTO)

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

        if fila_y + 5 >= zona.shape[0]:
            respuestas[str(i+1)] = "BLANCO"
            continue

        intensidades = []

        for letra in opciones:
            col_x = int(col_offsets[letra] * ancho)
            w = int(ancho_celda * ancho)
            h = int(alto_celda * alto)

            x_end = min(col_x + w, zona.shape[1])
            y_end = min(fila_y + h, zona.shape[0])

            celda = zona[fila_y:y_end, col_x:x_end]

            if celda.size == 0:
                intensidades.append(0)
                continue

            negro = (np.sum(celda) / celda.size) * 100
            intensidades.append(negro)

        marcadas = sum(v > 20 for v in intensidades)

        if marcadas == 0:
            respuestas[str(i+1)] = "BLANCO"
        elif marcadas > 1:
            respuestas[str(i+1)] = "DOBLE"
        else:
            idx = np.argmax(intensidades)
            respuestas[str(i+1)] = opciones[idx]

    return jsonify({
        "ok": True,
        "codigo": codigo,   # 🔥 AHORA SÍ DEVUELVE EL CÓDIGO
        "respuestas": respuestas
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
