# 
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode
import numpy as np
import os
import json

app = Flask(__name__)

# Cargar plantilla layout
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

# ---------------------------------------------------------
# 1. ENDPOINT PRINCIPAL: LECTURA OMR
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 2. PANEL: VER PLANTILLA SOBRE FONDO BLANCO
# ---------------------------------------------------------
@app.route("/omr/ver_plantilla")
def ver_plantilla():
    with open("plantilla_layout.json", "r") as f:
        plantilla = json.load(f)

    img = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(img)

    colores = {"A": "red", "B": "blue", "C": "green", "D": "orange"}

    for pregunta in plantilla["preguntas"]:
        num = pregunta["numero"]
        for op in pregunta["opciones"]:
            letra = op["letra"]
            x1, y1, x2, y2 = op["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=colores[letra], width=3)
            draw.text((x1, y1 - 12), f"{num}{letra}", fill=colores[letra])

    ruta = "plantilla_preview.png"
    img.save(ruta)
    return send_file(ruta, mimetype="image/png")


# ---------------------------------------------------------
# 3. PANEL: VER PLANTILLA SOBRE UNA IMAGEN REAL
# ---------------------------------------------------------
@app.route("/omr/ver_plantilla_sobre", methods=["POST"])
def ver_plantilla_sobre():
    if "file" not in request.files:
        return "Falta archivo", 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    draw = ImageDraw.Draw(img)

    with open("plantilla_layout.json", "r") as f:
        plantilla = json.load(f)

    colores = {"A": "red", "B": "blue", "C": "green", "D": "orange"}

    for pregunta in plantilla["preguntas"]:
        num = pregunta["numero"]
        for op in pregunta["opciones"]:
            letra = op["letra"]
            x1, y1, x2, y2 = op["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=colores[letra], width=3)
            draw.text((x1, y1 - 12), f"{num}{letra}", fill=colores[letra])

    ruta = "plantilla_sobre.png"
    img.save(ruta)
    return send_file(ruta, mimetype="image/png")


# ---------------------------------------------------------
# 4. PANEL HTML INTERACTIVO
# ---------------------------------------------------------
@app.route("/omr/panel")
def panel_html():
    return """
<!DOCTYPE html>
<html lang='es'>
<head>
<meta charset='UTF-8'>
<title>Panel OMR</title>
<style>
body { font-family: Arial; margin: 20px; }
h2 { margin-bottom: 10px; }
#preview { max-width: 100%; border: 1px solid #ccc; margin-top: 20px; }
.container { display: flex; gap: 40px; }
.block { width: 45%; }
input[type=file] { margin-top: 10px; }
button { padding: 10px 20px; margin-top: 10px; cursor: pointer; }
</style>
</head>
<body>

<h2>🧩 Panel visual OMR</h2>

<div class="container">

    <div class="block">
        <h3>Ver plantilla sobre fondo blanco</h3>
        <button onclick="verPlantilla()">Mostrar plantilla</button>
    </div>

    <div class="block">
        <h3>Ver plantilla sobre una imagen real</h3>
        <input type="file" id="imagen" accept="image/*">
        <button onclick="subirImagen()">Subir y ver</button>
    </div>

</div>

<img id="preview" src="" alt="Vista previa">

<script>
function verPlantilla() {
    document.getElementById("preview").src = "/omr/ver_plantilla?" + Date.now();
}

function subirImagen() {
    let file = document.getElementById("imagen").files[0];
    if (!file) {
        alert("Selecciona una imagen primero");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/omr/ver_plantilla_sobre", {
        method: "POST",
        body: formData
    })
    .then(r => r.blob())
    .then(blob => {
        document.getElementById("preview").src = URL.createObjectURL(blob);
    });
}
</script>

</body>
</html>
"""


# ---------------------------------------------------------
# ARRANQUE DEL SERVIDOR
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
