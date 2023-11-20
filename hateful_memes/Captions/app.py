import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# TODO - LOAD CAPTIONS MODEL
model = ...

app = FastAPI()


class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""

    ...


@app.post("/generate_captions")
async def predict(data: InputData):
    # TODO - IMPLEMENT THIS ENDPOINT
    try:
        features = ...

        caption = model.predict(features)

        return {"caption": caption}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
