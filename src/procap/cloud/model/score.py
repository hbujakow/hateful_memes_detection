import os
import json
import mlflow
from io import StringIO
from mlflow.pyfunc.scoring_server import infer_and_parse_json_input, predictions_to_json


def init():
    global model
    global input_schema

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.pyfunc.load_model(model_path)
    input_schema = model.metadata.get_input_schema()


def run(raw_data: str):
    """
    Performs inference on the given data in JSON format using the trained ML model.
    """
    json_data = json.loads(raw_data)
    if "input_data" not in json_data.keys():
        raise Exception("Request must contain a top level key named 'input_data'")

    serving_input = json.dumps(json_data["input_data"])
    data = infer_and_parse_json_input(serving_input, input_schema)
    prediction_logits = model(data)

    result = StringIO()
    predictions_to_json(prediction_logits, result)
    return result.getvalue()
