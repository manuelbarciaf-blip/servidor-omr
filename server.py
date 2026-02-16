import cv2
import numpy as np
import json
import sys
from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode
import os
import tempfile

app = Flask(__name__)

# =========================
# LECTOR DE BARCODE (MEJORADO)
# =========================
def leer_barcode(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray)

        for barcode in barcodes:
            codigo = barcode.data.decode("utf-8")
            if "EXAM" in codigo:
                return codigo
        return None
    except:
        return None

# =========================
# PARSEAR CODIGO REAL
# EXAM007ALU030FECHA09022026
# =========================
def parsear_codigo(codigo):
    datos = {
        "id_examen": None,
        "id_alumno": None,
        "fecha": None
    }

    if not codigo:
        return datos

    try:
        if "EXAM" in codigo and "ALU" in codigo:
            datos["id_examen"] = int(codigo.split("EXAM")[1].split("ALU")[0])
            datos["id_alumno"] = int(codigo.split("ALU")[1].split("FECHA")[0])

        if "FECHA" in codigo:
            datos["fecha"] = codigo.split("FECHA")[1]

    except:
        pass

    return datos

# =========================
# DETECTOR OMR PROFESIONAL (PLANTILLA FIJA)
# =========================
def detectar_respuestas(img):
    h, w = img.shape[:2]

    # ZONA CENTRAL donde SIEMPRE están tus burbujas (según tus imágenes)
    zona = img[int(h*0.30):int(h*0.85), int(w*0.15):int(w*0.85)]

    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Buscar burbujas
    contornos, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    burbujas = []
    for c in contornos:
        area = cv2.contourArea(c)

        # Ajustado a tus burbujas reales (centrales y rellenas)
        if 300 < area < 3000:
            x, y, bw, bh = cv2.boundingRect(c)
            ratio = bw / float(bh)

            # Filtrar solo círculos (burbujas)
            if 0.6 < ratio < 1.4:
                burbujas.append((x, y, bw, bh))

    # Ordenar por filas (clave para 30 preguntas en 2 líneas)
    burbujas = sorted(burbujas, key=lambda b: (b[1], b[0]))

    respuestas = []
    opciones = ['A','B','C','D']

    filas = []
    fila_actual = []
    last_y = None

    # Agrupar por filas reales (tus 2 líneas de 15)
    for b in burbujas:
        x, y, bw, bh = b

        if last_y is None:
            last_y = y

        if abs(y - last_y) > 25:
            if fila_actual:
                filas.append(sorted(fila_actual, key=lambda k: k[0]))
            fila_actual = [b]
            last_y = y
        else:
            fila_actual.append(b)

    if fila_actual:
        filas.append(sorted(fila_actual, key=lambda k: k[0]))

    # Cada pregunta = grupo de 4 burbujas (A B C D)
    for fila in filas:
        for i in range(0, len(fila), 4):
            grupo = fila[i:i+4]
            if len(grupo) < 4:
                continue

            rellenos = []
            for (x, y, bw, bh) in grupo:
                roi = thresh[y:y+bh, x:x+bw]
                pixeles = cv2.countNonZero(roi)
                rellenos.append(pixeles)

            max_idx = np.argmax(rellenos)
            max_val = rellenos[max_idx]

            # Umbral para detectar si está marcada
            if max_val < 150:
                respuestas.append("BLANCO")
            else:
                respuestas.append(opciones[max_idx])

    return respuestas

# =========================
# DIBUJAR CORRECCIÓN VISUAL
# =========================
def dibujar_correccion(img, respuestas):
    salida = img.copy()
    y = 50
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, r in enumerate(respuestas):
        cv2.putText(
            salida,
            f"{i+1}:{r}",
            (50, y),
            font,
            0.6,
            (0, 0, 255),
            2
        )
        y += 25

    return salida

# =========================
# ENDPOINT PARA TU PHP (/omr/leer)
# =========================
@app.route("/omr/leer", methods=["POST"])
def omr_leer():
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No se envió archivo"})

    file = request.files['file']

    # Archivo temporal (importante en Render)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    file.save(temp.name)

    img = cv2.imread(temp.name)

    if img is None:
        return jsonify({"ok": False, "error": "Imagen inválida"})

    # 1️⃣ Leer código de barras (PRIORIDAD MÁXIMA)
    codigo = leer_barcode(img)
    datos = parsear_codigo(codigo)

    # 2️⃣ Detectar respuestas REALES (no fijo a 20)
    respuestas = detectar_respuestas(img)

    # 3️⃣ Imagen corregida
    corregida = dibujar_correccion(img, respuestas)
    ruta_corregida = temp.name.replace(".jpg", "_corregido.jpg")
    cv2.imwrite(ruta_corregida, corregida)

    os.remove(temp.name)

    return jsonify({
        "ok": True,
        "barcode": codigo,
        "datos": datos,
        "num_preguntas_detectadas": len(respuestas),
        "respuestas": respuestas,
        "imagen_corregida": os.path.basename(ruta_corregida)
    })

# =========================
# ARRANQUE RENDER (MUY IMPORTANTE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
