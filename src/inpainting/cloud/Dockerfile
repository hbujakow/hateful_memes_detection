FROM python:3.10

WORKDIR /app

COPY ../. /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8089

CMD ["uvicorn", "app:app", "--reload", "--port", "8089", "--host", "0.0.0.0"]
