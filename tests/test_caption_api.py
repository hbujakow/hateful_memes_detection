import pytest
import base64
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import time
import concurrent.futures


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


def send_caption_request(sample_image: Image.Image) -> requests.Response:
    """
    Sends a POST request to the inpaint API.
    Returns:
        Dict: API response consisting of encoded image and extracted text.
    """
    sample_image.save(BUFFER, format="PNG")
    image_bytes = BUFFER.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image": encoded_image}

    response = requests.post(CAPTION_API_URL + "/generate_captions", json=payload)

    return response


def test_captions_ok(sample_image: Image.Image):
    """
    Test captions API.
    Args:
        sample_image (PIL.Image.Image): Sample image.
    """
    response = send_caption_request(sample_image)
    assert response.status_code == 200
    assert isinstance(response.json()["caption"], str)


def test_captions_4xx():
    response = requests.post(
        CAPTION_API_URL + "/generate_captions", json={"image": 123}
    )
    assert response.status_code >= 400 and response.status_code < 500
    response2 = requests.post(
        CAPTION_API_URL + "/generate_captions", json={"imageee": "123"}
    )
    assert response2.status_code >= 400 and response2.status_code < 500


def test_api_response_time(
    sample_image: Image.Image, acceptable_response_time: float = 10.0
):
    """
    Tests if the API reponds within acceptable response time.
    Args:
        sample_image (PIL.Image.Image): Sample image.
        acceptable_response_time (float): Acceptable response time in seconds.
    """
    start_time = time.time()
    _ = send_caption_request(sample_image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    assert elapsed_time < acceptable_response_time


@pytest.mark.parametrize("num_requests", [5, 10])
def test_concurrent_requests(sample_image: Image.Image, num_requests: int):
    """
    Tests if the API can handle concurrent requests.
    Args:
        sample_image (PIL.Image.Image): Sample image.
        num_requests (int): Number of concurrent requests.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        future_to_request = {
            executor.submit(send_caption_request, sample_image): i
            for i in range(num_requests)
        }

        results = [
            future.result()
            for future in concurrent.futures.as_completed(future_to_request)
        ]

    assert all(result.status_code == 200 for result in results)
