import base64
import os
import time
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

PROCAP_API_URL = "http://127.0.0.1:8087/predict"
CAPTION_API_URL = "http://127.0.0.1:8088/generate_captions"
INPAINT_API_URL = "http://127.0.0.1:8089/inpaint"

st.set_page_config(layout="wide")

buffered = BytesIO()


def call_inpaint_image_api(image, extract_text=True):
    """
    Inpaints the image. Optionally, returns the text extracted from image with OCR.
    """
    image.save("image.png")
    with open("image.png", "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {"image": encoded_image}
    os.remove("image.png")

    try:
        response = requests.post(INPAINT_API_URL, json=payload)
    except Exception as e:
        return f"Error: {e}"

    if response.status_code == 200:
        result = response.json()
        image = Image.open(BytesIO(base64.b64decode((result["image"]))))
        text = result["text"]

        result = {"image": image, "text": text} if extract_text else {"image": image}

        return result
    return f"Error: {response.status_code} - {response.text}"


def call_caption_api(inpainted_image):
    inpainted_image.save("inpainted_image.png")
    with open("inpainted_image.png", "rb") as inpainted_image:
        encoded_image = base64.b64encode(inpainted_image.read()).decode("utf-8")

    payload = {"image": encoded_image}
    os.remove("inpainted_image.png")
    try:
        response = requests.post(CAPTION_API_URL, json=payload)
    except Exception as e:
        return f"Error: {e}"

    if response.status_code == 200:
        result = response.json()
        return result["caption"]
    return f"Error: {response.status_code} - {response.text}"


def call_procap_api(caption):
    payload = {"text": caption}

    try:
        response = requests.post(PROCAP_API_URL, json=payload)
    except Exception as e:
        return f"Error: {e}"

    if response.status_code == 200:
        result = response.json()
        return result
    return f"Error: {response.status_code} - {response.text}"


def main():
    st.title("Detecting harmful and offensive content in memes")
    st.write("Upload a meme image file for classification")

    # Upload jpg data
    uploaded_file = st.file_uploader("Upload meme image", type=["png", "jpg"])

    if uploaded_file is not None:
        start = time.time()
        meme_image = Image.open(uploaded_file)
        st.write("Meme to analyze:")
        st.image(meme_image)

        # Data preprocessing
        st.header("Preprocessing the image")

        # call all apis, firstly inpaint, then caption then procap
        inpainted_results = call_inpaint_image_api(meme_image)
        inpainted_image = inpainted_results["image"]
        text = inpainted_results["text"]
        st.write(f"Extracted text: {text}")

        st.write("Inpainting complete.")
        st.image(inpainted_image)

        caption = call_caption_api(inpainted_image)
        st.write("Captioning complete.")

        input_text = text + " . " + "It was <mask>" + " . " + caption + " . </s>"
        st.write("Text to be analyzed:")
        st.write(input_text)

        procap_results = call_procap_api(input_text)
        st.write("Procap complete.")
        st.write("Results:")
        st.write(procap_results["prediction"])
        st.write(procap_results["probability"])

        st.write("Model inference complete.")
        time_taken = time.time() - start
        st.write(f"Time taken: {time_taken} seconds")


if __name__ == "__main__":
    main()
