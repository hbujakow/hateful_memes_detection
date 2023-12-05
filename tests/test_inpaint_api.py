import pytest
import base64
import concurrent.futures
import time
from io import BytesIO
import requests
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define the base URL for the API
INPAINT_API_URL = "http://127.0.0.1:8089"
BUFFER = BytesIO()


@pytest.fixture
def sample_image():
    """
    Creates sample image with text.
    Returns:
        PIL.Image.Image: Sample image.
    """
    # create a sample image
    sample_image = Image.new("RGB", (200, 200), color="white")

    # add sample text to the image
    drawer = ImageDraw.Draw(sample_image)
    font = ImageFont.truetype("arial.ttf", size=16)
    drawer.text((10, 10), "Sample meme text", fill=(0, 0, 0), font=font, size=100)

    return sample_image


def send_inpaint_request(sample_image):
    """
    Sends a POST request to the inpaint API.
    Args:
        sample_image (PIL.Image.Image): Sample image.
    Returns:
        Dict: API response consisting of encoded image and extracted text.
    """
    assert isinstance(sample_image, Image.Image)
    sample_image.save(BUFFER, format="PNG")
    image_bytes = BUFFER.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image": encoded_image}
    response = requests.post(INPAINT_API_URL + "/inpaint", json=payload)

    return response


def test_inpaint_api_response_ok(sample_image):
    """
    Tests inpaint API if it returns the correct format of response and status code.
    """
    response = send_inpaint_request(sample_image)
    assert response.status_code == 200
    assert isinstance(response.json()["image"], str)
    assert isinstance(response.json()["text"], str)


def test_inpaint_api_4xx():
    """
    Tests if API returns the appropriate status code.
    """
    response = requests.post(INPAINT_API_URL + "/inpaint", json={"image": "123"})
    assert response.status_code >= 400 and response.status_code < 500
    response2 = requests.post(INPAINT_API_URL + "/inpaint", json={"imageee": "123"})
    assert response2.status_code >= 400 and response2.status_code < 500
    response3 = requests.post(INPAINT_API_URL + "/inpaint", json={"image": 123})
    assert response3.status_code >= 400 and response3.status_code < 500


def test_inpaint_api_image_inpainting(sample_image):
    """
    Tests if the image is inpainted correctly by the inpaint API,
    i.e. there is no text on the inpainted image.
    """

    response = send_inpaint_request(sample_image)

    decoded_image = base64.b64decode(response.json()["image"])
    assert decoded_image is not None

    inpainted_image = Image.open(BytesIO(decoded_image))
    assert isinstance(inpainted_image, Image.Image)

    text_on_inpainted_image = easyocr.Reader(["en"]).readtext(np.array(inpainted_image))
    assert text_on_inpainted_image == []


def test_inpaint_api_extracted_text(sample_image):
    """
    Tests if the extracted text from the image is correct.
    """

    response = send_inpaint_request(sample_image)
    assert response.json()["text"] == "Sample meme text"


@pytest.mark.parametrize("num_requests", [5, 10])
def test_concurrent_requests(sample_image, num_requests):
    """
    Tests if the inpaint API can handle multiple concurrent requests.
    Args:
        sample_image (PIL.Image.Image): Sample image.
        num_requests (int): Number of concurrent requests.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        future_to_request = {
            executor.submit(send_inpaint_request, sample_image): i
            for i in range(num_requests)
        }

        results = [
            future.result()
            for future in concurrent.futures.as_completed(future_to_request)
        ]

    assert all(result.status_code == 200 for result in results)


def test_api_response_time(sample_image, acceptable_response_time=5.0):
    """
    Tests if the response time of the API is within the acceptable threshold.
    Args:
        sample_image (PIL.Image.Image): Sample image.
        acceptable_response_time (float): Threshold for acceptable response time.
    """

    start_time = time.time()
    response = send_inpaint_request(sample_image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    assert elapsed_time < acceptable_response_time
