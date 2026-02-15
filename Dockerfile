FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para OpenCV y pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Puerto que usa Render
ENV PORT=8080

# Ejecutar con gunicorn (mejor que flask run)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "server:app"]
