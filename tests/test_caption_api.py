import pytest
import base64
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont

CAPTION_API_URL = "http://127.0.0.1:8088"
BUFFER = BytesIO()


@pytest.fixture
def sample_image():
    """
    Creates sample image with text.
    Returns:
        PIL.Image.Image: Sample image.
    """
    sample_image = Image.new("RGB", (200, 200), color="white")

    drawer = ImageDraw.Draw(sample_image)
    font = ImageFont.truetype("arial.ttf", size=16)
    drawer.text((10, 10), "Sample meme text", fill=(0, 0, 0), font=font, size=100)

    return sample_image

def send_caption_request(sample_image):
    """
    Sends a POST request to the inpaint API.
    Returns:
        Dict: API response consisting of encoded image and extracted text.
    """
    sample_image.save(BUFFER, format="PNG")
    image_bytes = BUFFER.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image": encoded_image}

    response = requests.post(CAPTION_API_URL+"/generate_captions", json=payload)

    return response

def test_captions_ok(sample_image):
    """
    Test captions API.
    Args:
        sample_image (PIL.Image.Image): Sample image.
    """
    response = send_caption_request(sample_image)
    assert response.status_code == 200
    assert isinstance(response.json()["caption"], str)

def test_captions_4xx():
    response = requests.post(CAPTION_API_URL+"/generate_captions", json={"image": 123})
    assert response.status_code >= 400 and response.status_code < 500
    response2 = requests.post(CAPTION_API_URL+"/generate_captions", json={"imageee": "123"})
    assert response2.status_code >= 400 and response2.status_code < 500
