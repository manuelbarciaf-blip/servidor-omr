FROM python:3.10-slim

# Dependencias del sistema para OpenCV + pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
