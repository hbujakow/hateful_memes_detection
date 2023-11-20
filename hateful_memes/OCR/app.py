import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# TODO - LOAD OCR MODEL
model = ...

app = FastAPI()


class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""

    ...


@app.post("/extract_text")
async def predict(data: InputData):
    # TODO - IMPLEMENT THIS ENDPOINT
    try:
        features = ...

        text = model.predict(features)

        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
