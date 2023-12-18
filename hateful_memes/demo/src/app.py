import base64
import os
import time
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
import config

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

BUFFER = BytesIO()


def call_inpaint_image_api(image, extract_text=True):
    """
    Inpaints the image. Optionally, returns the text extracted from image with OCR.
    """

    image.save(BUFFER, format="PNG")
    image_bytes = BUFFER.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image": encoded_image}

    response = requests.post(config.INPAINT_API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        image = Image.open(BytesIO(base64.b64decode((result["image"]))))
        text = result["text"]

        result = {"image": image, "text": text} if extract_text else {"image": image}

        return result
    raise Exception(f"Error: {response.status_code} - {response.text}")


def call_caption_api(inpainted_image):
    """
    Generates captions for the inpainted image.
    """
    inpainted_image.save(BUFFER, format="PNG")
    image_bytes = BUFFER.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image": encoded_image}
    try:
        response = requests.post(config.CAPTION_API_URL, json=payload)
    except Exception as e:
        return f"Error: {e}"

    if response.status_code == 200:
        result = response.json()
        return result["caption"]
    raise Exception(f"Error: {response.status_code} - {response.text}")


def call_procap_api(caption):
    """
    Classifies the caption as harmful or not.
    """
    payload = {"text": caption}

    try:
        response = requests.post(config.PROCAP_API_URL, json=payload)
    except Exception as e:
        return f"Error: {e}"

    if response.status_code == 200:
        result = response.json()
        return result
    raise Exception(f"Error: {response.status_code} - {response.text}")


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

        try:
            with st.spinner("Inpainting image..."):
                inpainted_results = call_inpaint_image_api(meme_image)
        except Exception:
            col1.write("Inpainting failed. Please try again.")
            return
        else:
            col1.write("Inpainting complete.")

        inpainted_image = inpainted_results["image"]
        text = inpainted_results["text"]
        try:
            with st.spinner("Generating captions..."):
                caption = call_caption_api(inpainted_image)
        except Exception:
            col1.write("Captioning failed. Please try again.")
            return

        col1.write("Captioning complete.")

        input_text = text + " . " + "It was <mask>" + " . " + caption + " . </s>"

        try:
            with st.spinner("Classifying meme..."):
                procap_results = call_procap_api(input_text)
        except Exception:
            col1.write("Classification failed. Please try again.")
            return
        col1.write("Classifying complete.")

        col2.markdown("#### Results:")

        col2.markdown("Extracted text:")
        col2.markdown(f"```\n{text}\n```")

        col2.markdown("Text to be analyzed:")
        col2.markdown(f"```\n{input_text}\n```")

        col2.write("##### Classification:")

        result = "Hateful" if procap_results["prediction"] == 1 else "Not hateful"
        probability = (
            procap_results["probability"]
            if result == "Hateful"
            else 1 - procap_results["probability"]
        )
        col2.markdown(result)
        col2.write(f"with probability: {probability * 100:.2f}%")

        time_taken = time.time() - start
        col1.write(f"Time taken: {round(time_taken, 2)} seconds")


if __name__ == "__main__":
    main()
