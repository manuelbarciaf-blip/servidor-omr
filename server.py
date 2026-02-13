from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import json
import os

app = Flask(__name__)

with open("plantilla_layout.json", "r") as f:
    layout = json.load(f)

def detectar_hoja_por_marcas(img_gray):
    """
    Detecta las 4 marcas negras de las esquinas y corrige perspectiva
    """
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 2000:  # marcas negras grandes
            x,y,w,h = cv2.boundingRect(c)
            ratio = w/h if h != 0 else 0
            if 0.5 < ratio < 1.5:
                candidatos.append((x,y,w,h))

    if len(candidatos) < 4:
        return img_gray  # fallback si no detecta marcas

    # Ordenar por posición (esquinas)
    candidatos = sorted(candidatos, key=lambda b: b[0]+b[1])

    pts = []
    for (x,y,w,h) in candidatos[:4]:
        pts.append([x+w//2, y+h//2])

    pts = np.array(pts, dtype="float32")

    # Orden: TL, TR, BL, BR
    pts = pts[np.argsort(pts[:,1])]
    top = pts[:2]
    bottom = pts[2:]

    top = top[np.argsort(top[:,0])]
    bottom = bottom[np.argsort(bottom[:,0])]

    rect = np.array([top[0], top[1], bottom[0], bottom[1]], dtype="float32")

    ancho = 1000
    alto = 1400

    dst = np.array([
        [0,0],
        [ancho,0],
        [0,alto],
        [ancho,alto]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_gray, M, (ancho, alto))

    return warped


@app.route("/omr/leer", methods=["POST"])
def leer_omr():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No se envió archivo"})

        file = request.files["file"]

        # Convertir imagen a OpenCV
        img_pil = Image.open(file.stream).convert("RGB")
        img_np = np.array(img_pil)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 🔧 CORRECCIÓN AUTOMÁTICA PARA MÓVIL
        hoja = detectar_hoja_por_marcas(img_gray)

        # Binarización optimizada para lápiz
        blur = cv2.GaussianBlur(hoja, (5,5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25, 10
        )

        ANCHO = 1000
        ALTO = 1400

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
            scores = []

            for letra in opciones:
                col_x = int(col_offsets[letra] * ancho)
                w = int(ancho_celda * ancho)
                h = int(alto_celda * alto)

                celda = zona[fila_y:fila_y+h, col_x:col_x+w]

                if celda.size == 0:
                    scores.append(0)
                    continue

                # % de píxeles negros (perfecto para lápiz)
                score = cv2.countNonZero(celda) / (w*h)
                scores.append(score)

            max_score = max(scores)
            marcadas = sum(s > 0.15 for s in scores)

            if max_score < 0.10:
                respuestas[str(i+1)] = "BLANCO"
            elif marcadas > 1:
                respuestas[str(i+1)] = "DOBLE"
            else:
                idx = int(np.argmax(scores))
                respuestas[str(i+1)] = opciones[idx]

        return jsonify({
            "ok": True,
            "respuestas": respuestas
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
