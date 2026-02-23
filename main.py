from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omr import procesar_omr

app = FastAPI()

@app.post("/corregir_omr")
async def corregir_omr(request: Request):
    try:
        binario = await request.body()
        resultado = procesar_omr(binario)
        return JSONResponse(resultado)
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
