FROM dustynv/l4t-pytorch:r35.3.1

WORKDIR /app

COPY . .

CMD ["python3", "main.py"]
