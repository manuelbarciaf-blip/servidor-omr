from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from omr import procesar_omr
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

@app.post("/corregir_omr")
async def corregir_omr(imagen: UploadFile = File(...)):
    try:
        # Leer binario real de la imagen
        binario = await imagen.read()

        if not binario:
            logging.warning("Imagen vacía o no recibida")
            return JSONResponse({"ok": False, "error": "No se recibió imagen"}, status_code=400)

        resultado = procesar_omr(binario)

        if not resultado.get("ok"):
            logging.error(f"Error OMR: {resultado.get('error')}")
        else:
            logging.info(f"OMR OK: Examen={resultado.get('id_examen')} Alumno={resultado.get('id_alumno')}")

        return JSONResponse(resultado)

    except Exception as e:
        logging.exception("Error inesperado")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"ok": True, "mensaje": "Servidor OMR activo"}
