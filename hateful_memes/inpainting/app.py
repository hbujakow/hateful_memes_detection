import base64
from io import BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from Inpainter import ImageConverter

app = FastAPI()
img_converter = ImageConverter()


class InputData(BaseModel):
    """Data model for input data to API"""

    image: str


@app.post("/inpaint")
async def predict(image: InputData):
    im = Image.open(BytesIO(base64.b64decode((image.image))))
    img_converter.upload_img(image = im)
    try:
        inpainted_image = img_converter.inpaint_image()
        buffered = BytesIO()
        inpainted_image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        text = img_converter.retrieve_text()
        return {"image": encoded_image, "text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
