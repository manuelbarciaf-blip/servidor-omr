from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
from pyzbar.pyzbar import decode
import json, os, re

app = Flask(__name__)

# Cargar layout base (posición de burbujas)
with open("plantilla_layout.json", "r") as f:
    layout_base = json.load(f)

def leer_codigo_barras(img):
    """Recorta la zona del código de barras y lee CODE128"""
    W, H = img.size
    crop_w = int(W * 0.40)
    crop_h = int(H * 0.15)
    crop_x = W - crop_w - 20
    crop_y = 20
    zona = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    barcodes = decode(zona)
    if not barcodes:
        return None
    return barcodes[0].data.decode("utf-8")

def detectar_burbujas(img, layout, num_preguntas):
    """Detecta burbujas marcadas en la imagen"""
    W, H = img.size
    img = img.convert("L")  # escala de grises
    img_np = np.array(img)

    # umbral adaptativo
    thresh = img_np < np.mean(img_np) * 0.8

    respuestas = {}
    lay = layout["layout"]
    x0 = int(lay["zona_respuestas"]["x0_rel"] * W)
    y0 = int(lay["zona_respuestas"]["y0_rel"] * H)
    ancho = int(lay["zona_respuestas"]["ancho_rel"] * W)
    alto = int(lay["zona_respuestas"]["alto_rel"] * H)

    offset_y = lay["filas"]["offset_y_rel"]
    alto_celda = lay["filas"]["alto_celda_rel"]
    ancho_celda = lay["casilla"]["ancho_rel"]

    col_offsets = {
        "A": lay["columnas"]["A"]["offset_x_rel"],
        "B": lay["columnas"]["B"]["offset_x_rel"],
        "C": lay["columnas"]["C"]["offset_x_rel"],
        "D": lay["columnas"]["D"]["offset_x_rel"]
    }

    opciones = layout["opciones"]

    for i in range(num_preguntas):
        fila_y = int(i * offset_y * alto)
        intensidades = []
        for letra in opciones:
            col_x = int(col_offsets[letra] * ancho)
            w = int(ancho_celda * ancho)
            h = int(alto_celda * alto)
            x_end = min(col_x + w, thresh.shape[1])
            y_end = min(fila_y + h, thresh.shape[0])
            celda = thresh[fila_y:y_end, col_x:x_end]
            if celda.size == 0:
                intensidades.append(0)
                continue
            negro = (np.sum(celda) / celda.size) * 100
            intensidades.append(negro)
        marcadas = sum(v > 15 for v in intensidades)
        if marcadas == 0:
            respuestas[str(i+1)] = "BLANCO"
        elif marcadas > 1:
            respuestas[str(i+1)] = "DOBLE"
        else:
            idx = np.argmax(intensidades)
            respuestas[str(i+1)] = opciones[idx]

    return respuestas

@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400
    file = request.files["file"]
    try:
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img)
    except:
        return jsonify({"ok": False, "error": "Imagen inválida"}), 400

    # normalizar tamaño
    img = img.resize((1000, 1400))

    codigo = leer_codigo_barras(img)
    if not codigo:
        codigo = "DESCONOCIDO"

    # Detectar número de preguntas dinámicamente
    num_preguntas = 20
    m = re.match(r'EXAM\d+ALU(\d+)', codigo)
    if m:
        # extrae el número de preguntas del código o usa plantilla de examen
        id_examen = int(re.search(r'EXAM(\d+)', codigo).group(1))
        ruta_plantilla = f'plantillas/examen_{id_examen:03d}.json'
        if os.path.exists(ruta_plantilla):
            with open(ruta_plantilla, 'r') as f:
                plant = json.load(f)
            num_preguntas = len(plant.get("respuestas_correctas", []))
        else:
            num_preguntas = int(m.group(1))  # fallback
    else:
        num_preguntas = 30

    respuestas = detectar_burbujas(img, layout_base, num_preguntas)

    return jsonify({
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "num_preguntas": num_preguntas
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
