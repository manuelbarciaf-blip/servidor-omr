FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV PYTHONUNBUFFERED=1
# Un proceso para no duplicar el consumo de memoria de OpenCV en la instancia Free.
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "150", "--access-logfile", "-", "--error-logfile", "-", "omr:app"]
