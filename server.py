from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
from pyzbar.pyzbar import decode
import json
import io

app = Flask(__name__)

# Cargar layout base
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

def leer_codigo_barras(img):
    """Recorta la zona superior derecha y lee CODE128"""
    W, H = img.size
    # Ajustar recorte según tu plantilla
    crop_w = int(W * 0.35)
    crop_h = int(H * 0.12)
    crop_x = W - crop_w - 20
    crop_y = 20
    zona = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    zona = zona.convert("L")  # escala de grises
    # Incrementar contraste simple
    np_zona = np.array(zona)
    np_zona = np.where(np_zona>200, 255, np_zona)
    zona = Image.fromarray(np_zona)
    barcodes = decode(zona)
    if not barcodes:
        return None
    return barcodes[0].data.decode("utf-8")

def detectar_burbujas(img, layout, num_preguntas):
    """Detecta burbujas marcadas"""
    W, H = img.size
    img = img.convert("L")
    img_np = np.array(img)
    thresh = img_np < 160  # simple umbral para móviles

    respuestas = {}
    lay = layout["layout"]
    x0 = int(lay["zona_respuestas"]["x0_rel"]*W)
    y0 = int(lay["zona_respuestas"]["y0_rel"]*H)
    ancho = int(lay["zona_respuestas"]["ancho_rel"]*W)
    alto = int(lay["zona_respuestas"]["alto_rel"]*H)

    offset_y = lay["filas"]["offset_y_rel"]
    alto_celda = lay["filas"]["alto_celda_rel"]
    ancho_celda = lay["casilla"]["ancho_rel"]

    col_offsets = {k: lay["columnas"][k]["offset_x_rel"] for k in layout["opciones"]}
    opciones = layout["opciones"]

    for i in range(num_preguntas):
        fila_y = int(i * offset_y * alto)
        intensidades = []
        for letra in opciones:
            col_x = int(col_offsets[letra]*ancho)
            w = int(ancho_celda*ancho)
            h = int(alto_celda*alto)
            x_end = min(col_x + w, thresh.shape[1])
            y_end = min(fila_y + h, thresh.shape[0])
            celda = thresh[fila_y:y_end, col_x:x_end]
            if celda.size == 0:
                intensidades.append(0)
                continue
            negro = (np.sum(celda)/celda.size)*100
            intensidades.append(negro)
        marcadas = sum(v>15 for v in intensidades)
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

    img = img.resize((1000, 1400))  # normalizar tamaño
    codigo = leer_codigo_barras(img)
    if not codigo:
        codigo = "DESCONOCIDO"

    # Detectar número de preguntas según código: EXAM###ALU## -> alumno indica num preguntas
    import re
    num_preguntas = 20
    m = re.search(r'EXAM\d+ALU(\d+)', codigo)
    if m:
        num_preguntas = int(m.group(1))
        if num_preguntas < 20 or num_preguntas > 60:
            num_preguntas = 20
    respuestas = detectar_burbujas(img, layout, num_preguntas)

    return jsonify({
        "ok": True,
        "codigo": codigo,
        "respuestas": respuestas,
        "num_preguntas": num_preguntas
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
