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
model_name = "distilroberta-base"
states_path = "/home2/faculty/mgalkowski/memes_analysis/hateful_memes/procap/models/pbm_1111_mem/221120231605.pth"

model = PromptHateModel(
    label_words=label_words, max_length=max_length, model_name=model_name
)

states = torch.load(states_path)

model.load_state_dict(states, strict=False)
model.eval()

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

class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""

    text: str


@app.post("/predict")
async def predict(data: InputData):
    logits = model(data.text).cuda()
    norm_logits = F.softmax(logits, dim=-1)[:, 1].unsqueeze(-1)
    probability = round(norm_logits.item(), 4)
    predicted_class = torch.where(norm_logits > 0.5, 1, 0).item()

    return {"prediction": predicted_class, "probability": probability}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
