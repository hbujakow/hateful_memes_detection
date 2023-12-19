import base64
from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from lavis.models import load_model_and_preprocess
from PIL import Image
from pydantic import BaseModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
)

app = FastAPI()

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

person_categories = {
    "race": "what is the race of the person in the image?",
    "gender": "what is the gender of the person in the image?",
    "country": "which country does the person in the image come from?",
    "religion": "what is the religion of the person in the image?",
}

animal_categories = {
    "animal": "what animal is in the image?",
}


def generate_prompt_result(model, vis_processors, device, im, ques):
    image = vis_processors["eval"](im).float().unsqueeze(0).to(device)
    ans = model.generate(
        {"image": image, "prompt": ("Question: %s Answer:" % (ques))},
        length_penalty=3.0,
    )
    return ans[0]


class InputData(BaseModel):
    """Data model for input data to API"""

    image: str

@app.post("/generate_captions")
async def predict(image: InputData):
    im = Image.open(BytesIO(base64.b64decode((image.image))))
    captions = {}
    person_on_img = generate_prompt_result(
        model, vis_processors, device, im, "is there a person in the image?"
    )
    is_person = person_on_img.startswith("yes")

    if is_person:
        for category, questions in person_categories.items():
            caption = generate_prompt_result(
                model, vis_processors, device, im, questions
            )
            captions[category] = caption

        not_disabled = generate_prompt_result(
            model,
            vis_processors,
            device,
            im,
            "are there disabled people in the image?",
        )
        if not not_disabled:
            captions["valid_disabled"] = "there is a disabled person"

    animal_on_img = generate_prompt_result(
        model, vis_processors, device, im, "is there an animal in the image?"
    )
    is_animal = animal_on_img.startswith("yes")

    if is_animal:
        caption = generate_prompt_result(
            model, vis_processors, device, im, "what animal is in the image?"
        )
        captions["animal"] = caption

    response = ""
    for category in "race,gender,country,animal,valid_disable,religion".split(","):
        if category not in captions:
            continue
        response += captions[category] + " . "
    return {"caption": response}



@app.post("/generic_caption")
async def generate_generic_caption(image: InputData):
    im = Image.open(BytesIO(base64.b64decode((image.image))))
    generic_caption = generate_prompt_result(
        model, vis_processors, device, im, "describe briefly what is in the image"
    )

    return {"generic_caption": generic_caption}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
