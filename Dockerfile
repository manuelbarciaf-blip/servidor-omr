FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para pyzbar
RUN apt-get update && apt-get install -y libzbar0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
