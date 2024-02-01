FROM python:3.10

WORKDIR /procap

COPY . /procap

RUN pip install -r requirements.txt

EXPOSE 8088

CMD ["uvicorn", "procap:app", "--reload", "--port", "8088", "--host", "0.0.0.0"]
