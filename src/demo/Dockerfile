FROM python:3.10.0-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

ENV PROCAP_API_URL=http://memes-classification-api.azurewebsites.net/api/predict
ENV CAPTION_API_URL=http://memes-captioning-api.azurewebsites.net/api/generate_captions
ENV INPAINT_API_URL=http://memes-inpainting-api.azurewebsites.net/api/inpaint

CMD ["streamlit", "run", "app.py"]
