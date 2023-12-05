import base64
from io import BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from Inpainter import ImageConverter
from PIL import Image
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()
img_converter = ImageConverter()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(ValueError)
async def http_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": exc.args[0]},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


class InputData(BaseModel):
    """Data model for input data to API"""

    image: str


@app.post("/inpaint")
async def inpaint(image: InputData):
    im = Image.open(BytesIO(base64.b64decode((image.image))))
    if not im.format:
        raise ValueError("Invalid image format. Please upload a valid image.")
    img_converter.upload_img(image=im)
    inpainted_image = img_converter.inpaint_image()
    buffered = BytesIO()
    inpainted_image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    text = img_converter.retrieve_text()
    return {"image": encoded_image, "text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
