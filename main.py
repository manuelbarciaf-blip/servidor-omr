from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from omr import procesar_omr
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.post("/corregir_omr")
async def corregir_omr(imagen: UploadFile = File(...)):
    try:
        binario = await imagen.read()

        if not binario:
            return JSONResponse({"ok": False, "error": "Imagen vacía"}, status_code=400)

        resultado = procesar_omr(binario)
        return JSONResponse(resultado)

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"ok": True, "mensaje": "Servidor OMR activo"}
