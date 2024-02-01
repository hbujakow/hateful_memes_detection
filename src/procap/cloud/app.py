import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

load_dotenv()

subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_ML_WORKSPACE")

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
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


class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""

    text: str


@app.post("/predict")
async def predict(data: InputData):
    request_file = {"input_data": data.text}
    logits = ml_client.online_endpoints.invoke(
        endpoint_name="hateful-memes-classifier",
        request_file=request_file,
    )
    norm_logits = F.softmax(logits, dim=-1)[:, 1].unsqueeze(-1)
    probability = round(norm_logits.item(), 4)
    predicted_class = torch.where(norm_logits > 0.5, 1, 0).item()

    return {"prediction": predicted_class, "probability": probability}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
