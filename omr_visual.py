from PIL import Image, ImageDraw, ImageOps
import numpy as np
import json
from pyzbar.pyzbar import decode
import os

# Cargar layout
with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

opciones = layout["opciones"]

# Carpeta con imágenes escaneadas
carpeta_examenes = "omr_uploads/2026-02-14"

def leer_codigo_barras(img):
    W, H = img.size
    zona = img.crop((W-420, 20, W-20, 210))
    barcodes = decode(zona)
    return barcodes[0].data.decode("utf-8") if barcodes else "DESCONOCIDO"

def detectar_burbujas(img, layout, num_preguntas):
    W, H = img.size
    img = img.convert("L")
    img_np = np.array(img)
    thresh = img_np < np.mean(img_np) * 0.8

    lay = layout["layout"]
    x0 = int(lay["zona_respuestas"]["x0_rel"] * W)
    y0 = int(lay["zona_respuestas"]["y0_rel"] * H)
    ancho = int(lay["zona_respuestas"]["ancho_rel"] * W)
    alto = int(lay["zona_respuestas"]["alto_rel"] * H)

    offset_y = lay["filas"]["offset_y_rel"]
    alto_celda = lay["filas"]["alto_celda_rel"]
    ancho_celda = lay["casilla"]["ancho_rel"]
    col_offsets = {letra: lay["columnas"][letra]["offset_x_rel"] for letra in opciones}

    respuestas = {}
    coords = {}

    for i in range(num_preguntas):
        fila_y = int(i * offset_y * alto)
        intensidades = []
        celda_coords = []

        for letra in opciones:
            col_x = int(col_offsets[letra] * ancho)
            w = int(ancho_celda * ancho)
            h = int(alto_celda * alto)
            x_end = min(col_x + w, thresh.shape[1])
            y_end = min(fila_y + h, thresh.shape[0])
            celda = thresh[fila_y:y_end, col_x:x_end]
            if celda.size == 0:
                intensidades.append(0)
            else:
                intensidades.append((np.sum(celda)/celda.size)*100)
            celda_coords.append((x0+col_x, y0+fila_y, x0+col_x+w, y0+fila_y+h))

        marcadas = sum(v > 15 for v in intensidades)
        if marcadas == 0:
            respuestas[str(i+1)] = "BLANCO"
        elif marcadas > 1:
            respuestas[str(i+1)] = "DOBLE"
        else:
            idx = np.argmax(intensidades)
            respuestas[str(i+1)] = opciones[idx]

        coords[str(i+1)] = celda_coords

    return respuestas, coords

# Procesar cada examen
for archivo in os.listdir(carpeta_examenes):
    if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    ruta = os.path.join(carpeta_examenes, archivo)
    img = Image.open(ruta)
    img = ImageOps.exif_transpose(img)
    img = img.resize((1000,1400))

    codigo = leer_codigo_barras(img)
    import re
    m = re.match(r'EXAM\d+ALU(\d+)', codigo)
    num_preguntas = int(m.group(1)) if m else 30

    respuestas, coords = detectar_burbujas(img, layout, num_preguntas)

    # Dibujar burbujas
    draw = ImageDraw.Draw(img)
    for q, opc_coords in coords.items():
        for idx, (x1, y1, x2, y2) in enumerate(opc_coords):
            color = "grey"
            letra = opciones[idx]
            if respuestas[q] == letra:
                color = "yellow" if respuestas[q] == "DOBLE" else "green"
            draw.ellipse([x1,y1,x2,y2], outline=color, width=3)

    salida = os.path.join(carpeta_examenes, f"vis_{archivo}")
    img.save(salida)
    print(f"✔ Generada visualización: {salida} ({num_preguntas} preguntas)")
