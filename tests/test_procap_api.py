import pytest
import base64
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import time
import concurrent.futures

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

def test_api_response_time(sample_text, acceptable_response_time=10.0):

    start_time = time.time()
    response = send_procap_request(sample_text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    assert elapsed_time < acceptable_response_time

@pytest.mark.parametrize("num_requests", [5, 10])
def test_concurrent_requests(sample_text, num_requests):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        future_to_request = {
            executor.submit(send_procap_request, sample_text): i
            for i in range(num_requests)
        }

        results = [
            future.result()
            for future in concurrent.futures.as_completed(future_to_request)
        ]

    assert all(result.status_code == 200 for result in results)
