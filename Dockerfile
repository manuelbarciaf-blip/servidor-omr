FROM python:3.10

RUN apt-get update && apt-get install -y libzbar0

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "server.py"]
