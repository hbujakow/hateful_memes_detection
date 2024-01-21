# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV PROCAP_API_URL=http://memes-classification-api.azurewebsites.net/api/predict
ENV CAPTION_API_URL=http://memes-captioning-api.azurewebsites.net/api/generate_captions
ENV INPAINT_API_URL=http://memes-inpainting-api.azurewebsites.net/api/inpaint

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
