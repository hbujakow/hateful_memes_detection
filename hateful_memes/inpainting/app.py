import base64
from io import BytesIO
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Inpainter import ImageConverter
from PIL import Image
import numpy as np

app = FastAPI()


class InputData(BaseModel):
    """Data model for input data to API"""
    image: str


@app.post("/inpaint")
async def predict(image: InputData):
    im = Image.open(BytesIO(base64.b64decode((image.image))))
    model = ImageConverter(image = im)
    try:
        inpainted_image = model.inpaint_image()
        buffered = BytesIO()
        inpainted_image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        text = model.retrieve_text()
        return {"image": encoded_image, "text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
