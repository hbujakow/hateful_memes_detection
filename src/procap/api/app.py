import sys

sys.path.append("../")

import torch
import torch.nn.functional as F
import uvicorn
from architecture.pbm import PromptHateModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

label_words = ["good", "bad"]
max_length = 447
model_name = "roberta-large"
states_path = "/home2/faculty/mgalkowski/memes_analysis/hateful_memes/procap/models/1111_mem/20240107_1425_roberta_large.pth"

model = PromptHateModel(
    label_words=label_words, max_length=max_length, model_name=model_name
)

states = torch.load(states_path)

model.load_state_dict(states, strict=False)
model.eval()

app = FastAPI()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles HTTP exceptions and returns a JSON response.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(ValueError)
async def http_exception_handler(request: Request, exc: ValueError):
    """
    Handles value errors and returns a JSON response.
    """
    return JSONResponse(
        status_code=400,
        content={"message": exc.args[0]},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handles generic exceptions and returns a JSON response.
    """
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""

    text: str


@app.post("/predict")
async def predict(data: InputData):
    """
    API endpoint to predict the probability of hate speech in a given prompt.
    """
    logits = model(data.text).cuda()
    norm_logits = F.softmax(logits, dim=-1)[:, 1].unsqueeze(-1)
    probability = round(norm_logits.item(), 4)
    predicted_class = torch.where(norm_logits > 0.5, 1, 0).item()

    return {"prediction": predicted_class, "probability": probability}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
