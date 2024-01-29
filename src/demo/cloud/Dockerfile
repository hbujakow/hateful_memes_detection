FROM python:3.10

WORKDIR /app

COPY ../. /app

RUN pip install -r requirements.txt

EXPOSE 8501

# TODO: Add environment variables for API URLs based on container apps
# ENV PROCAP_API_URL=
# ENV CAPTION_API_URL=
# ENV INPAINT_API_URL=

CMD ["streamlit", "run", "app.py"]
