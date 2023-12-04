import pytest
import base64
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont

PROCAP_API_URL = "http://127.0.0.1:8087"
BUFFER = BytesIO()


@pytest.fixture
def sample_text():
    return "This is a test text. It was <mask>"

def send_procap_request(sample_text):
    payload = {"text": sample_text}
    response = requests.post(PROCAP_API_URL+"/predict", json=payload)
    return response

def test_procap_ok(sample_text):
    response = send_procap_request(sample_text)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], int)
    assert isinstance(response.json()["probability"], float)

def test_procap_4xx():
    response = send_procap_request(123)
    assert response.status_code >= 400 and response.status_code < 500
    response2 = requests.post(PROCAP_API_URL+"/predict", json={"textt": "This is a test text."})
    assert response2.status_code >= 400 and response2.status_code < 500
    response3 = requests.post(PROCAP_API_URL+"/predict", json={"text": "This is a test text."})
    assert response3.status_code >= 400 and response3.status_code < 500
