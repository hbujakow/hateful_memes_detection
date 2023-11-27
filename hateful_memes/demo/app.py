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

st.set_page_config(
    layout="wide",
    page_title="Meme hatefulness classifier",
    page_icon="🤔",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)

buffered = BytesIO()


def call_inpaint_image_api(image, extract_text=True):
    """
    Inpaints the image. Optionally, returns the text extracted from image with OCR.
    """
    image.save("temp_img.png")
    with open("temp_img.png", "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {"image": encoded_image}
    os.remove("temp_img.png")

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
    """
    Generates captions for the inpainted image.
    """
    inpainted_image.save("inpainted_temp_img.png")
    with open("inpainted_temp_img.png", "rb") as inpainted_image:
        encoded_image = base64.b64encode(inpainted_image.read()).decode("utf-8")

    payload = {"image": encoded_image}
    os.remove("inpainted_temp_img.png")
    try:
        response = requests.post(CAPTION_API_URL, json=payload)
    except Exception as e:
        return f"Error: {e}"

    if response.status_code == 200:
        result = response.json()
        return result["caption"]
    return f"Error: {response.status_code} - {response.text}"


def call_procap_api(caption):
    """
    Classifies the caption as harmful or not.
    """
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

    uploaded_file = st.file_uploader(
        "Upload a meme image file for classification", type=["png", "jpg"]
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        start = time.time()

        meme_image = Image.open(uploaded_file)
        col1.markdown("#### Uploaded meme:")
        col1.image(meme_image, use_column_width=True)

        with st.spinner("Inpainting image..."):
            inpainted_results = call_inpaint_image_api(meme_image)
        col1.write("Inpainting complete.")

        inpainted_image = inpainted_results["image"]
        text = inpainted_results["text"]

        with st.spinner("Generating captions..."):
            caption = call_caption_api(inpainted_image)
        col1.write("Captioning complete.")

        input_text = text + " . " + "It was <mask>" + " . " + caption + " . </s>"

        with st.spinner("Classifying meme..."):
            procap_results = call_procap_api(input_text)

        col1.write("Classifying complete.")

        col2.markdown("#### Results:")

        col2.markdown("Extracted text:")
        col2.markdown(f"```\n{text}\n```")

        col2.markdown("Text to be analyzed:")
        col2.markdown(f"```\n{input_text}\n```")

        col2.write("##### Classification:")

        result = "Hateful" if procap_results["prediction"] == 1 else "Not hateful"
        col2.markdown(result)
        col2.write(f"with probability: {procap_results['probability']}")

        time_taken = time.time() - start
        col1.write(f"Time taken: {round(time_taken, 2)} seconds")


if __name__ == "__main__":
    main()
