from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omr import procesar_omr
import logging

# -------------------------------
# CONFIGURAR LOGGING
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="Servidor OMR", description="API para procesar OMR desde PHP", version="1.0")

# -------------------------------
# ENDPOINT PRINCIPAL
# -------------------------------
@app.post("/corregir_omr")
async def corregir_omr(request: Request):
    try:
        # Leer binario de la imagen enviada
        binario = await request.body()
        if not binario:
            logging.warning("No se recibió ningún archivo")
            return JSONResponse({"ok": False, "error": "No se recibió ningún archivo"}, status_code=400)

        resultado = procesar_omr(binario)

        if not resultado.get("ok"):
            logging.error(f"Error al procesar OMR: {resultado.get('error')}")
        else:
            logging.info(f"OMR procesado correctamente: Examen={resultado.get('id_examen')} Alumno={resultado.get('id_alumno')}")

        return JSONResponse(resultado)

    except Exception as e:
        logging.exception("Error inesperado al procesar OMR")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# -------------------------------
# ENDPOINT DE PRUEBA (opcional)
# -------------------------------
@app.get("/")
async def root():
    return {"ok": True, "mensaje": "Servidor OMR activo"}
