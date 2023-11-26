# PROCAP microservice

### Setup

1. Create environment and install dependencies

```bash
conda create -n api_procap python=3.10
conda activate api_procap
conda env update --file environment.yaml
```

2. Run the server

```bash
uvicorn app:app --reload --port 8088 --host 0.0.0.0
```

### How to make request?

```python
import requests


def generate_caption(sample):
    api_url = "http://127.0.0.1:8087/predict"

    payload = {"text": sample}

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code} - {response.text}"


sample = "notice how the kenyan skidmark has been silent about the mueller report .  It was <mask> . president barack obama standing in front of two american flags . he is african-american and he is speaking in front of two american flags in front of a podium in front of  . the man in the image is president barack obama of the united states he is standing in front of two american flags . u.s. president barack obama is in front of two american flags in front of a podium in front of . he is a christian and he is standing in front of a u.s. flag and two american flags . </s>"

# sample needs to be in a format: OCR text + template + generic caption + procap caption + </s> [everything seperated by ' . ']

print(generate_caption(sample)["prediction"])
print(generate_caption(sample)["probability"])
```
