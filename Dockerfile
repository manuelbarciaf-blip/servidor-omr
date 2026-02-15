FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libzbar0 \
    libzbar-dev \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "server.py"]
