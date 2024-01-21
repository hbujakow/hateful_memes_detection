import sys
import torch
import torch.nn.functional as F
import json
import azure.functions as func
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append("../architecture/")

from architecture.pbm import PromptHateModel

label_words = ["good", "bad"]
max_length = 447
model_name = "roberta-large"
states_path = "/home2/faculty/mgalkowski/memes_analysis/hateful_memes/procap/models/1111_mem/20240107_1425_roberta_large.pth"  # Replace with your local path

model = PromptHateModel(
    label_words=label_words, max_length=max_length, model_name=model_name
)

states = torch.load(states_path)
model.load_state_dict(states, strict=False)
model.eval()

app = FastAPI()


def http_exception_handler(exc: Exception) -> func.HttpResponse:
    status_code = 500
    if isinstance(exc, func.HttpException):
        status_code = exc.status_code
    return func.HttpResponse(
        json.dumps({"message": str(exc)}),
        status_code=status_code,
        mimetype="application/json",
    )


class InputData(BaseModel):
    text: str


def predict(data: InputData):
    logits = model(data.text).cuda()
    norm_logits = F.softmax(logits, dim=-1)[:, 1].unsqueeze(-1)
    probability = round(norm_logits.item(), 4)
    predicted_class = torch.where(norm_logits > 0.5, 1, 0).item()

    return {"prediction": predicted_class, "probability": probability}


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        data = InputData(**req_body)
        result = predict(data)
        return func.HttpResponse(json.dumps(result), mimetype="application/json")
    except Exception as e:
        return http_exception_handler(e)
