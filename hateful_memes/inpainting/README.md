# Inpainting microservice

This microservice serves the purpose of inpainting the meme image (i.e. removing the text from the image and filling in the gap to obtain a clear image) and extracting the text using Optical Character Recognition.

### Folder structure

```
├───model
│   ├───networks.py
│   ├───__init__.py
│
├───pretrained
│       README.md
│
├───app.py
├───environment.yml
├───Inpainter.py
├───README.md

```

`model` dir contains the logic of the model inpainting the images - [source](https://github.com/nipponjo/deepfillv2-pytorch/blob/master/model/networks.py).

`pretrained` dir contains the instruction on where to get the pretrained model weights from.



### Setup

1. Create environment and install dependencies
```bash
conda create -n inpainting python=3.10
conda activate inpainting
conda env update --file environment.yaml
```

2. Run the server
```bash
uvicorn app:app --reload --port 8089 --host 0.0.0.0
```

### How to make request?

```python
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8089/inpaint"

def inpaint_image(image_path, extract_text = False):
    """
    Inpaints the image. Optionally, returns the text extracted from image with OCR.
    """
    with open(image_path, 'rb') as image_binary:
        encoded_image = base64.b64encode(image_binary.read()).decode('utf-8')

    payload = {"image": encoded_image}

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        image = Image.open(BytesIO(base64.b64decode((result["image"]))))
        text = result["text"]

        return (image, text) if extract_text else image
    else:
        return f"Error: {response.status_code} - {response.text}"
```
