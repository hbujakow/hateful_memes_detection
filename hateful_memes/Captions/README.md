# Captions microservice

### Setup

1. Create environment and install dependencies
```bash
conda create -n captions python=3.10
conda activate captions
conda env update --file environment.yaml
```

2. Run the server
```bash
uvicorn app:app --reload
```

### How to make request?

```python
import requests
import base64

def generate_caption(image_path):
    with open(image_path, 'rb') as image_binary:
        encoded_image = base64.b64encode(image_binary.read()).decode('utf-8')

    api_url = "http://127.0.0.1:8088/generate_captions"

    payload = {"image": encoded_image}

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["caption"]
    else:
        return f"Error: {response.status_code} - {response.text}"
```