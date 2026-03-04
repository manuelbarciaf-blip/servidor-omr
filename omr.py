from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

A4_W, A4_H = 2480, 3508
OPCIONES = ["A","B","C","D"]

OMR_REGION = {"y0":650,"y1":3000,"x0":780,"x1":1450}

MAX_FILAS_POR_HOJA = 30

UMBRAL_VACIO = 0.055
UMBRAL_DOBLE_RATIO = 0.78
UMBRAL_DOBLE_ABS = 0.040


# =============================
# NORMALIZAR A4
# =============================
def normalizar_a4_con_marcas(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,th = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV)

    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    h,w = gray.shape[:2]

    candidatos = []

    for c in cnts:

        area = cv2.contourArea(c)

        if area < 2500:
            continue

        x,y,bw,bh = cv2.boundingRect(c)

        ratio = bw/float(bh)

        if 0.65 < ratio < 1.35:

            cx = x + bw/2
            cy = y + bh/2

            candidatos.append((cx,cy))

    if len(candidatos) < 4:

        return cv2.resize(img,(A4_W,A4_H))

    pts = np.array(candidatos,dtype="float32")

    rect = np.zeros((4,2),dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([[0,0],[A4_W,0],[A4_W,A4_H],[0,A4_H]],dtype="float32")

    M = cv2.getPerspectiveTransform(rect,dst)

    return cv2.warpPerspective(img,M,(A4_W,A4_H))


# =============================
# LEER QR
# =============================
def leer_qr(img):

    det = cv2.QRCodeDetector()

    data,_,_ = det.detectAndDecode(img)

    if data:
        return data.strip()

    crop = img[0:1200,0:1200]

    crop = cv2.resize(crop,None,fx=2.5,fy=2.5)

    data,_,_ = det.detectAndDecode(crop)

    return data.strip() if data else None


# =============================
# BINARIZACIÓN
# =============================
def binarizar(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8))

    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray,(5,5),0)

    th = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,10
    )

    kernel = np.ones((3,3),np.uint8)

    th = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)
    th = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel)

    return th


# =============================
# NUEVO DETECTOR DE BURBUJAS
# =============================
def detectar_respuestas(zona_bin,filas,debug):

    h,w = zona_bin.shape

    alto_fila = int(h / MAX_FILAS_POR_HOJA)
    ancho_op = int(w / 4)

    respuestas = []

    for i in range(filas):

        y0 = i * alto_fila
        y1 = (i+1) * alto_fila

        fila = zona_bin[y0:y1,:]

        scores = []

        for j in range(4):

            x0 = j * ancho_op
            x1 = (j+1) * ancho_op

            celda = fila[:,x0:x1]

            if celda.size == 0:

                scores.append(0)
                continue

            # buscar círculos dentro de la celda
            circles = cv2.HoughCircles(
                celda,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=15,
                minRadius=8,
                maxRadius=35
            )

            tinta = 0

            if circles is not None:

                circles = np.uint16(np.around(circles))

                for c in circles[0,:]:

                    mask = np.zeros(celda.shape,dtype="uint8")

                    cv2.circle(mask,(c[0],c[1]),c[2],255,-1)

                    tinta = cv2.countNonZero(
                        cv2.bitwise_and(celda,celda,mask=mask)
                    )

            else:

                tinta = cv2.countNonZero(celda)

            dens = tinta / float(celda.size)

            scores.append(dens)

        max_d = max(scores)

        idx = scores.index(max_d)

        segundo = sorted(scores,reverse=True)[1]

        if max_d < UMBRAL_VACIO:

            resp = ""

        elif (segundo > UMBRAL_DOBLE_ABS) and (segundo > max_d * UMBRAL_DOBLE_RATIO):

            resp = "X"

        else:

            resp = OPCIONES[idx]

        respuestas.append(resp)

        # debug dibujo
        for j in range(4):

            x0 = OMR_REGION["x0"] + j*ancho_op
            yA = OMR_REGION["y0"] + y0
            x1 = x0 + ancho_op
            yB = yA + alto_fila

            cv2.rectangle(debug,(x0,yA),(x1,yB),(0,255,0),2)

            if resp and OPCIONES[j] == resp:

                cv2.rectangle(debug,(x0,yA),(x1,yB),(0,0,255),3)

    return respuestas,debug


# =============================
# PIPELINE
# =============================
def procesar_omr(binario):

    npimg = np.frombuffer(binario,np.uint8)

    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    if img is None:
        return {"ok":False,"error":"Imagen inválida"}

    img_a4 = normalizar_a4_con_marcas(img)

    codigo = leer_qr(img_a4)

    if not codigo:
        return {"ok":False,"error":"QR no detectado"}

    partes = codigo.split("|")

    if len(partes) < 3:
        return {"ok":False,"error":"QR inválido"}

    id_examen = int(partes[0])
    id_alumno = int(partes[1])
    num_preguntas = int(partes[2])

    pagina = int(partes[3]) if len(partes) >= 4 else 1

    th = binarizar(img_a4)

    zona_bin = th[OMR_REGION["y0"]:OMR_REGION["y1"],OMR_REGION["x0"]:OMR_REGION["x1"]]

    debug = img_a4.copy()

    filas = min(num_preguntas,MAX_FILAS_POR_HOJA)

    respuestas_lista,debug = detectar_respuestas(zona_bin,filas,debug)

    respuestas = {}

    for i,r in enumerate(respuestas_lista,start=1):

        respuestas[str(i)] = r

    _,buff = cv2.imencode(".jpg",debug,[int(cv2.IMWRITE_JPEG_QUALITY),85])

    debug_image = base64.b64encode(buff).decode("utf-8")

    return {
        "ok":True,
        "codigo":codigo,
        "id_examen":id_examen,
        "id_alumno":id_alumno,
        "num_preguntas":num_preguntas,
        "pagina":pagina,
        "respuestas":respuestas,
        "debug_image":debug_image
    }


@app.route("/corregir_omr",methods=["POST"])
def corregir_omr():

    if "imagen" not in request.files:

        return jsonify({"ok":False,"error":"Falta imagen"}),400

    binario = request.files["imagen"].read()

    res = procesar_omr(binario)

    return jsonify(res)


@app.route("/")
def home():

    return "Servidor OMR funcionando ✅"


if __name__ == "__main__":

    port = int(os.environ.get("PORT",8080))

    app.run(host="0.0.0.0",port=port)
