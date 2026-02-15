import cv2
import numpy as np
import json
import sys
from pyzbar.pyzbar import decode
import re

def leer_barcode(img):
    # Buscar en la franja superior (más robusto)
    h = img.shape[0]
    zona = img[0:int(h*0.35), :]
    
    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    
    # Mejorar contraste (clave para móviles)
    gray = cv2.equalizeHist(gray)
    
    barcodes = decode(gray)
    
    for barcode in barcodes:
        texto = barcode.data.decode("utf-8")
        return texto
    
    return None

def parsear_codigo(codigo):
    examen = re.search(r'EXAM(\d+)', codigo)
    alumno = re.search(r'ALU(\d+)', codigo)
    fecha = re.search(r'FECHA(\d+)', codigo)
    
    return {
        "id_examen": examen.group(1) if examen else None,
        "id_alumno": alumno.group(1) if alumno else None,
        "fecha": fecha.group(1) if fecha else None
    }

def detectar_esquinas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cuadrados = []
    for c in contours:
        area = cv2.contourArea(c)
        if 1000 < area < 10000:
            x,y,w,h = cv2.boundingRect(c)
            if abs(w-h) < 20:
                cuadrados.append((x,y,w,h))
    
    return cuadrados

def detectar_respuestas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    burbujas = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if 300 < area < 2000:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = w / float(h)
            
            if 0.7 < ratio < 1.3:
                burbujas.append((x,y,w,h))
    
    burbujas = sorted(burbujas, key=lambda b: (b[1], b[0]))
    
    respuestas = []
    opciones = ['A','B','C','D']
    
    for i in range(0, len(burbujas), 4):
        grupo = burbujas[i:i+4]
        if len(grupo) < 4:
            continue
        
        grupo = sorted(grupo, key=lambda g: g[0])
        
        valores = []
        for (x,y,w,h) in grupo:
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            valores.append(total)
        
        indice = np.argmax(valores)
        respuestas.append(opciones[indice])
    
    return respuestas

def dibujar_correccion(img, respuestas):
    salida = img.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 200
    
    for i, r in enumerate(respuestas):
        cv2.putText(salida, f"{i+1}:{r}", (50, y), font, 0.6, (0,0,255), 2)
        y += 25
    
    return salida

if __name__ == "__main__":
    ruta = sys.argv[1]
    
    img = cv2.imread(ruta)
    
    # 1 Leer barcode (OBLIGATORIO)
    codigo = leer_barcode(img)
    
    if codigo:
        datos = parsear_codigo(codigo)
    else:
        datos = {"id_examen": None, "id_alumno": None, "fecha": None}
    
    # 2 Detectar respuestas reales (no fijo a 20)
    respuestas = detectar_respuestas(img)
    
    # 3 Imagen corregida
    corregida = dibujar_correccion(img, respuestas)
    ruta_corregida = ruta.replace(".jpg", "_corregido.jpg")
    cv2.imwrite(ruta_corregida, corregida)
    
    resultado = {
        "barcode": codigo,
        "datos": datos,
        "num_preguntas_detectadas": len(respuestas),
        "respuestas": respuestas,
        "imagen_corregida": ruta_corregida
    }
    
    print(json.dumps(resultado))
