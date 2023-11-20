import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# TODO - LOAD INPAINTING MODEL
model = ...

app = FastAPI()


class InputData(BaseModel):
    """Use this data model to parse the input data JSON request body."""

    ...


@app.post("/inpaint")
async def predict(data: InputData):
    # TODO - IMPLEMENT THIS ENDPOINT
    # MAYBE IMAGE NEED TO BE ENCODED IN BASE64 FOR TRANSMISSION
    try:
        features = ...

        image = model.predict(features)

        return {"image": image}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
