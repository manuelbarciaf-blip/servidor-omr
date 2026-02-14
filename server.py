from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from pyzbar.pyzbar import decode
import re, json

app = Flask(__name__)

with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

def leer_codigo_barras(img):
    """Intenta leer el código de barras en toda la imagen con autocontraste y grises"""
    img = ImageOps.exif_transpose(img)
    img = ImageOps.autocontrast(img)  # aumenta contraste
    img_gray = img.convert("L")
    barcodes = decode(img_gray)
    if barcodes:
        return barcodes[0].data.decode("utf-8")

    # fallback: esquina superior derecha
    W, H = img.size
    crop_w, crop_h = int(W*0.40), int(H*0.20)
    crop_x, crop_y = W - crop_w - 20, 10
    zona = img_gray.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    barcodes = decode(zona)
    if barcodes:
        return barcodes[0].data.decode("utf-8")
    return None

def detectar_burbujas(img, layout, num_preguntas):
    """Detecta burbujas marcadas y devuelve coordenadas"""
    W, H = img.size
    img_gray = img.convert("L")
    img_np = np.array(img_gray)
    thresh = img_np < np.mean(img_np)*0.8

    respuestas = {}
    coords = {}
    lay = layout["layout"]
    x0 = int(lay["zona_respuestas"]["x0_rel"] * W)
    y0 = int(lay["zona_respuestas"]["y0_rel"] * H)
    ancho = int(lay["zona_respuestas"]["ancho_rel"] * W)
    alto = int(lay["zona_respuestas"]["alto_rel"] * H)
    offset_y = lay["filas"]["offset_y_rel"]
    alto_celda = lay["filas"]["alto_celda_rel"]
    ancho_celda = lay["casilla"]["ancho_rel"]
    col_offsets = {letra: lay["columnas"][letra]["offset_x_rel"] for letra in layout["opciones"]}
    opciones = layout["opciones"]

    for i in range(num_preguntas):
        fila_y = int(i * offset_y * alto)
        intensidades = []
        celda_coords = []
        for letra in opciones:
            col_x = int(col_offsets[letra]*ancho)
            w, h = int(ancho_celda*ancho), int(alto_celda*alto)
            x_end = min(col_x + w, thresh.shape[1])
            y_end = min(fila_y + h, thresh.shape[0])
            celda = thresh[fila_y:y_end, col_x:x_end]
            negro = (np.sum(celda)/celda.size*100) if celda.size>0 else 0
            intensidades.append(negro)
            celda_coords.append((x0+col_x, y0+fila_y, x0+col_x+w, y0+fila_y+h))
        marcadas = sum(v>15 for v in intensidades)
        if marcadas == 0:
            respuestas[str(i+1)] = "BLANCO"
        elif marcadas>1:
            respuestas[str(i+1)] = "DOBLE"
        else:
            idx = np.argmax(intensidades)
            respuestas[str(i+1)] = opciones[idx]
        coords[str(i+1)] = celda_coords
    return respuestas, coords

@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400
    file = request.files["file"]
    try:
        img = Image.open(file.stream)
    except:
        return jsonify({"ok": False, "error": "Imagen inválida"}), 400

    img = img.resize((1400, 1800))  # tamaño grande para móvil
    codigo = leer_codigo_barras(img) or "DESCONOCIDO"

    # Número de preguntas
    num_preguntas = 20
    m = re.match(r'EXAM\d+ALU(\d+)', codigo)
    if m:
        num_preguntas = int(m.group(1))
    else:
        # fallback por nombre de archivo
        filename = request.files['file'].filename
        m2 = re.search(r'_(\d+)\.jpg$', filename)
        if m2:
            num_preguntas = int(m2.group(1))
    num_preguntas = max(20, min(num_preguntas, 60))

    respuestas, coords = detectar_burbujas(img, layout, num_preguntas)

    return jsonify({
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "num_preguntas": num_preguntas,
        "coords": coords
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
