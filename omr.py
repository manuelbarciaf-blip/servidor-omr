import cv2
import numpy as np
import base64

# =====================================================
# CONFIGURACIÓN
# =====================================================

A4_W = 2480
A4_H = 3508

MAX_FILAS_POR_HOJA = 30
OPCIONES = ["A","B","C","D"]

OMR_REGION = {
    "y0":650,
    "y1":3000,
    "x0":780,
    "x1":1450
}

UMBRAL_VACIO = 0.05
UMBRAL_DOBLE = 0.80


# =====================================================
# NORMALIZAR HOJA A4 (CORRECCIÓN PERSPECTIVA)
# =====================================================

def normalizar_a4(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,th = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)

    contornos,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    puntos = []

    for c in contornos:

        area = cv2.contourArea(c)

        if area < 2000:
            continue

        x,y,w,h = cv2.boundingRect(c)

        ratio = w/float(h)

        if 0.6 < ratio < 1.4:

            puntos.append([x+w/2,y+h/2])

    if len(puntos) < 4:

        return cv2.resize(img,(A4_W,A4_H))

    pts = np.array(puntos,dtype="float32")

    rect = np.zeros((4,2),dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([
        [0,0],
        [A4_W,0],
        [A4_W,A4_H],
        [0,A4_H]
    ],dtype="float32")

    M = cv2.getPerspectiveTransform(rect,dst)

    warped = cv2.warpPerspective(img,M,(A4_W,A4_H))

    return warped


# =====================================================
# LECTOR QR
# =====================================================

def leer_qr(img):

    detector = cv2.QRCodeDetector()

    data,_,_ = detector.detectAndDecode(img)

    if data:
        return data.strip()

    # fallback zona superior izquierda

    crop = img[0:1200,0:1200]

    crop = cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)

    data,_,_ = detector.detectAndDecode(crop)

    if data:
        return data.strip()

    return None


# =====================================================
# BINARIZACIÓN ROBUSTA
# =====================================================

def binarizar(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    kernel = np.ones((3,3),np.uint8)

    th = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel,iterations=1)
    th = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel,iterations=2)

    return th


# =====================================================
# DETECTAR RESPUESTAS POR CONTORNOS
# =====================================================

def detectar_respuestas(zona_bin,zona_color,filas):

    contornos,_ = cv2.findContours(
        zona_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    burbujas = []

    for c in contornos:

        area = cv2.contourArea(c)

        if area < 200 or area > 6000:
            continue

        x,y,w,h = cv2.boundingRect(c)

        ratio = w/float(h)

        if 0.6 < ratio < 1.4:

            burbujas.append((x,y,w,h,c))

    if len(burbujas) < filas*4:

        return [],zona_color

    burbujas = sorted(burbujas,key=lambda b:b[1])

    respuestas = []

    for i in range(filas):

        fila = burbujas[i*4:(i+1)*4]

        fila = sorted(fila,key=lambda b:b[0])

        scores = []

        for (x,y,w,h,c) in fila:

            mask = np.zeros(zona_bin.shape,dtype="uint8")

            cv2.drawContours(mask,[c],-1,255,-1)

            tinta = cv2.countNonZero(cv2.bitwise_and(zona_bin,zona_bin,mask=mask))

            area = cv2.countNonZero(mask)

            ratio = tinta/float(area)

            scores.append(ratio)

        max_v = max(scores)
        idx = scores.index(max_v)

        segundo = sorted(scores,reverse=True)[1]

        if max_v < UMBRAL_VACIO:

            resp = ""

        elif segundo > max_v*UMBRAL_DOBLE:

            resp = "X"

        else:

            resp = OPCIONES[idx]

        respuestas.append(resp)

        # debug visual

        for j,(x,y,w,h,c) in enumerate(fila):

            color = (0,255,0)

            if resp != "" and OPCIONES[j] == resp:
                color = (0,0,255)

            cv2.rectangle(
                zona_color,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

    return respuestas,zona_color


# =====================================================
# CALCULAR FILAS SEGÚN QR
# =====================================================

def filas_hoja(num_preguntas,pagina):

    if pagina == 1:
        return min(num_preguntas,MAX_FILAS_POR_HOJA),0

    if num_preguntas <= MAX_FILAS_POR_HOJA:
        return 0,MAX_FILAS_POR_HOJA

    return num_preguntas-MAX_FILAS_POR_HOJA,MAX_FILAS_POR_HOJA


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================

def procesar_omr(binario):

    npimg = np.frombuffer(binario,np.uint8)

    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    if img is None:
        return {"ok":False,"error":"Imagen inválida"}

    img_a4 = normalizar_a4(img)

    codigo = leer_qr(img_a4)

    if not codigo:
        return {"ok":False,"error":"QR no detectado"}

    partes = codigo.split("|")

    if len(partes) < 3:
        return {"ok":False,"error":"QR inválido"}

    id_examen = int(partes[0])
    id_alumno = int(partes[1])
    num_preguntas = int(partes[2])

    pagina = 1

    if len(partes) >= 4:
        pagina = int(partes[3])

    binaria = binarizar(img_a4)

    zona_bin = binaria[
        OMR_REGION["y0"]:OMR_REGION["y1"],
        OMR_REGION["x0"]:OMR_REGION["x1"]
    ]

    zona_color = img_a4[
        OMR_REGION["y0"]:OMR_REGION["y1"],
        OMR_REGION["x0"]:OMR_REGION["x1"]
    ].copy()

    filas,offset = filas_hoja(num_preguntas,pagina)

    respuestas_lista,debug = detectar_respuestas(
        zona_bin,
        zona_color,
        filas
    )

    respuestas = {}

    for i,r in enumerate(respuestas_lista,start=1):
        respuestas[str(offset+i)] = r

    _,buffer = cv2.imencode(".jpg",debug)

    debug_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "ok":True,
        "codigo":codigo,
        "id_examen":id_examen,
        "id_alumno":id_alumno,
        "num_preguntas":num_preguntas,
        "pagina":pagina,
        "respuestas":respuestas,
        "debug_image":debug_base64
    }
