import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pymysql
import sys
import os

# ============================
# CONFIGURACIÓN
# ============================

DB_CONFIG = {
    "host": "localhost",
    "user": "mygabiasfue9",
    "password": "gabias_2025",
    "database": "rrhhclases",
    "cursorclass": pymysql.cursors.DictCursor
}

# Región QR (arriba derecha)
QR_REGION = {
    "x0": 0.70,
    "y0": 0.02,
    "x1": 0.98,
    "y1": 0.18
}

# Región burbujas (centro A4)
OMR_REGION = {
    "x0": 0.30,
    "y0": 0.18,
    "x1": 0.70,
    "y1": 0.92
}

OPCIONES = ["A", "B", "C", "D"]

UMBRAL_MARCA = 0.45
UMBRAL_DOBLE = 0.75
UMBRAL_VACIO = 0.15


# ============================
# PREPROCESADO ESTILO GRAVIC
# ============================

def preparar_imagen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    return thresh


# ============================
# LECTURA QR ROBUSTA
# ============================

def leer_qr(img):
    h, w = img.shape[:2]

    x0 = int(w * QR_REGION["x0"])
    y0 = int(h * QR_REGION["y0"])
    x1 = int(w * QR_REGION["x1"])
    y1 = int(h * QR_REGION["y1"])

    qr_crop = img[y0:y1, x0:x1]

    decoded = decode(qr_crop)

    if not decoded:
        return None

    return decoded[0].data.decode("utf-8")


# ============================
# CONEXIÓN BD
# ============================

def obtener_datos_examen(examen_id):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("SELECT num_preguntas FROM examenes WHERE id_examen=%s", (examen_id,))
    examen = cursor.fetchone()

    cursor.execute("SELECT numero, correcta FROM preguntas WHERE id_examen=%s ORDER BY numero", (examen_id,))
    preguntas = cursor.fetchall()

    conn.close()

    return examen["num_preguntas"], preguntas


# ============================
# DETECCIÓN DE BURBUJAS
# ============================

def corregir_hoja(img, num_preguntas, offset_pregunta=0):

    thresh = preparar_imagen(img)
    h, w = thresh.shape[:2]

    x0 = int(w * OMR_REGION["x0"])
    y0 = int(h * OMR_REGION["y0"])
    x1 = int(w * OMR_REGION["x1"])
    y1 = int(h * OMR_REGION["y1"])

    omr_crop = thresh[y0:y1, x0:x1]

    resultados = []

    altura = omr_crop.shape[0]
    fila_alto = altura / 20  # 20 preguntas por hoja

    for i in range(min(20, num_preguntas - offset_pregunta)):
        y_inicio = int(i * fila_alto)
        y_fin = int((i+1) * fila_alto)

        fila = omr_crop[y_inicio:y_fin, :]

        ancho = fila.shape[1]
        opcion_ancho = ancho // 4

        densidades = []

        for j in range(4):
            x_inicio = j * opcion_ancho
            x_fin = (j+1) * opcion_ancho

            burbuja = fila[:, x_inicio:x_fin]
            pixeles_negros = cv2.countNonZero(burbuja)
            total_pixeles = burbuja.size

            densidad = pixeles_negros / total_pixeles
            densidades.append(densidad)

        max_densidad = max(densidades)
        indices_altos = [idx for idx, val in enumerate(densidades) if val > UMBRAL_MARCA]

        if max_densidad < UMBRAL_VACIO:
            marcada = ""
        elif len(indices_altos) > 1:
            marcada = "X"
        elif max_densidad < UMBRAL_MARCA:
            marcada = "?"
        else:
            marcada = OPCIONES[densidades.index(max_densidad)]

        resultados.append({
            "numero": offset_pregunta + i + 1,
            "marcada": marcada,
            "densidades": densidades
        })

    return resultados


# ============================
# CORRECCIÓN COMPLETA
# ============================

def corregir_examen(imagenes):

    img1 = cv2.imread(imagenes[0])
    qr = leer_qr(img1)

    if not qr:
        print("❌ QR no detectado")
        return

    examen_id, alumno_id, fecha = qr.split("|")

    num_preguntas, preguntas_bd = obtener_datos_examen(examen_id)

    resultados = []

    # Hoja 1
    resultados += corregir_hoja(img1, num_preguntas, 0)

    # Hoja 2 si existe
    if num_preguntas > 20 and len(imagenes) > 1:
        img2 = cv2.imread(imagenes[1])
        resultados += corregir_hoja(img2, num_preguntas, 20)

    aciertos = 0

    for r in resultados:
        correcta = preguntas_bd[r["numero"]-1]["correcta"]

        if r["marcada"] == correcta:
            aciertos += 1

        print(f'{r["numero"]} | Correcta: {correcta} | Marcada: {r["marcada"]}')

    print("\n✅ Aciertos:", aciertos)
    print("📊 Nota:", round((aciertos/num_preguntas)*10,2))


# ============================
# EJECUCIÓN
# ============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python omr.py hoja1.jpg [hoja2.jpg]")
        sys.exit()

    imagenes = sys.argv[1:]
    corregir_examen(imagenes)
