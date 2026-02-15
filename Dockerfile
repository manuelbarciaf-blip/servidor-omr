# Imagen base compatible con pyzbar
FROM python:3.10-slim

# Instalar librerías necesarias para pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libzbar-dev \
    && apt-get clean

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Railway usa el puerto de la variable PORT
EXPOSE 8080

# Comando de arranque
CMD ["python", "server.py"]
