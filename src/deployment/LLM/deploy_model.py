from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv
import datetime
import os

load_dotenv()

subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_ML_WORKSPACE")


def main():
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    model_name = "vinai-tweet-hateful-memes"
    model_local_path = "src/procap/cloud/model_artifacts"
    model = ml_client.models.create_or_update(
        Model(name=model_name, path=model_local_path, type=AssetTypes.MLFLOW_MODEL)
    )

    endpoint_name = "hateful-memes-classifier"

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="An online endpoint to generate predictions for hateful memes.",
        auth_mode="key",
    )

    ml_client.begin_create_or_update(endpoint)

    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=model,
        instance_type="Standard_F4s_v2",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(blue_deployment)

    endpoint.traffic = {"blue": 100}

    ml_client.begin_create_or_update(endpoint).result()


if __name__ == "__main__":
    main()
