import sys
import base64
from io import BytesIO
import json
import azure.functions as func
from PIL import Image

sys.path.append("../architecture/")

from Inpainter import ImageConverter

img_converter = ImageConverter()


def http_exception_handler(exc: Exception) -> func.HttpResponse:
    status_code = 500
    if isinstance(exc, func.HttpException):
        status_code = exc.status_code
    return func.HttpResponse(
        json.dumps({"message": str(exc)}),
        status_code=status_code,
        mimetype="application/json",
    )


def process_image(image_data):
    return Image.open(BytesIO(base64.b64decode(image_data)))


def inpaint(im):
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


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        image_data = req_body.get("image", "")
        im = process_image(image_data)
        result = inpaint(im)
        return func.HttpResponse(json.dumps(result), mimetype="application/json")
    except Exception as e:
        return http_exception_handler(e)
