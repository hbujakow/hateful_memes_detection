FROM python:3.10

WORKDIR /app

COPY ../. /app

RUN pip install -r requirements.txt

EXPOSE 8088

CMD ["uvicorn", "app:app", "--reload", "--port", "8088", "--host", "0.0.0.0"]
