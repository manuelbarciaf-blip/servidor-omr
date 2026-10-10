"""Clases360 OMR: lectura conservadora. No corrige ni escribe calificaciones."""
import re
import cv2
import numpy as np

MM_TO_PX = 10.0  # hoja virtual de 2100 x 2970
W, H = 2100, 2970
LETTERS = 'ABCDEFGH'


def _decode_qr(image):
    detector = cv2.QRCodeDetector()
    for src in (image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            frame = src if rot is None else cv2.rotate(src, rot)
            data, _, _ = detector.detectAndDecode(frame)
            if data:
                return data.strip()
    return ''


def _align(image):
    """Exige cuatro marcas negras cuadradas; nunca acepta resize como alineación fiable."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ih, iw = gray.shape
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w < iw * .009 or h < ih * .009 or area < iw * ih * .00008:
            continue
        ratio = w / float(h)
        extent = area / float(w*h)
        if .65 < ratio < 1.4 and extent > .65 and w < iw*.10 and h < ih*.10:
            candidates.append((x+w/2, y+h/2, area))
    if len(candidates) < 4:
        raise ValueError('No se detectan las cuatro marcas de referencia')
    # Elegir una marca por cada cuadrante en las posiciones esperadas de la hoja.
    expected = [(24.5/210,39.5/297),(185.5/210,39.5/297),
                (185.5/210,255.5/297),(24.5/210,255.5/297)]
    used = set()
    points = []
    for fx, fy in expected:
        options = [(i, (c[0]/iw-fx)**2+(c[1]/ih-fy)**2) for i,c in enumerate(candidates) if i not in used]
        idx, dist = min(options, key=lambda p:p[1])
        if dist > .035:  # escaneo excesivamente recortado o marca equivocada
            raise ValueError('Las marcas no coinciden con la geometría OMR')
        used.add(idx)
        points.append(candidates[idx][:2])
    source = np.float32(points)
    target = np.float32([[245,395],[1855,395],[1855,2555],[245,2555]])
    transform = cv2.getPerspectiveTransform(source,target)
    warped = cv2.warpPerspective(image,transform,(W,H),borderValue=(255,255,255))
    return warped


def procesar_clases360(binario, filas=20, opciones=None):
    """Devuelve marcas y dudas; el servidor PHP debe validar token y tipos de pregunta."""
    if not 1 <= filas <= 30:
        return {'ok':False,'error':'Filas fuera de rango (1 a 30)'}
    if opciones is None:
        opciones = [4]*filas
    if len(opciones) != filas or any(not 2 <= int(n) <= 8 for n in opciones):
        return {'ok':False,'error':'Distribución de opciones no válida'}
    arr = np.frombuffer(binario,dtype=np.uint8)
    image = cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if image is None:
        return {'ok':False,'error':'Imagen no válida; convertir PDF a PNG/JPG antes de enviar'}
    token = _decode_qr(image)
    if not re.fullmatch(r'C360OMR1:[a-fA-F0-9]{32}',token):
        return {'ok':False,'error':'QR Clases360 ilegible o inválido','requiere_revision':True}
    try:
        aligned = _align(image)
    except ValueError as exc:
        return {'ok':False,'error':str(exc),'token':token.split(':',1)[1],'requiere_revision':True}
    gray = cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)
    results = []
    for i in range(filas):
        cy = round((84 + i*5.7)*10)
        scores = []
        for j in range(int(opciones[i])):
            cx = round((75+j*12)*10)
            # Interior del círculo, excluye la línea impresa (radio exterior 2.35 mm).
            r = 12
            roi = gray[cy-r:cy+r+1,cx-r:cx+r+1]
            yy,xx = np.ogrid[-r:r+1,-r:r+1]
            vals = roi[(xx*xx+yy*yy)<=r*r]
            scores.append(round(float(np.mean(vals < 120)),3))
        # Umbrales deliberadamente conservadores: requieren revisión hasta calibración.
        certain = [LETTERS[j] for j,s in enumerate(scores) if s >= .50]
        possible = [LETTERS[j] for j,s in enumerate(scores) if .12 <= s < .50]
        estado = 'revisar' if possible else ('marcada' if certain else 'blanco')
        results.append({'numero':i+1,'marcas':certain,'posibles':possible,
                        'estado':estado,'densidades':dict(zip(LETTERS,scores))})
    return {'ok':True,'token':token.split(':',1)[1],
            'respuestas':results,
            'requiere_revision':any(r['estado']=='revisar' for r in results),
            'advertencia':'Lectura experimental. No registrar notas sin validar calibración y tipos de pregunta.'}
