# Hateful Memes Detection application deployment on Azure

The deployment of the system is done on the Azure cloud by utilizing several Azure services to host the architecture.

## Cloud infrastructure

The concept is depicted on the diagram.

![Diagram](deployment_azure.png)

The architecture on cloud leverages several components:
* Large Language Model [vinai/bertweet-large](https://huggingface.co/vinai/bertweet-large) used for classification is hosted using Azure Machine Learning
* Each microservice is containerized and Docker images are pushed to Azure Container Registry
* Containers are managed by Azure Kubernetes Service

## Deployment

**Prerequisites**:
* Valid Azure subscription.
* Azure client installed (`az` cli).
* Docker Engine installed on your local computer. You can download and install it from [Docker website](https://docs.docker.com/engine/install/).
* Terraform installed on your machine. You can download and install it from [Terraform Downloads](https://www.terraform.io/downloads.html).
* Kubernetes client installed (`kubectl`). You can download and install it from [Kubernetes Tools](https://kubernetes.io/docs/tasks/tools/).

1. **Log in** to your Azure acount using Aure client commands:
```bash
az login
az account set --subscription "your-azure-subscription-id"
```

<!-- 2. **Create resource group and container registry** using Azure client command:
```bash
az group create --name memes-resource-group --location WestEurope
az acr create --resource-group memes_rg --name memescontainerregistry --sku Basic
``` -->


2. **Create** the cloud infrastructure using Terraform:
```bash
cd deployment # change dir to deployment folder.
terraform init
terraform import azurerm_resource_group.memes-resource-group /subscriptions/<your-azure-subscription-id>/resourceGroups/memes-resource-group
terraform plan
terraform apply
```

3. **Host** the LLM model by using the Azure Machine Learning Python SDK script (set appropriate env variables first!):
```bash
pip install -r deployment/requirements.txt
python deployment/deploy_model.py
```

4. **Build** each Docker image and **push** the container to Azure Container Registry using Docker commands.
*Disclaimer* the predefined images names are respectively:
* Streamlit web app demo - *hateful_memes_app*
* Inpainting API - *memes_inpainting_api*
* Captioning API - *memes_captioning_api*
* Classifying API - *memes_classifying_api*

```bash
docker login memescontainerregistry.azurecr.io
cd ../demo/ # navigate to each folder (captions, demo, inpainting, procap)
docker build -t memescontainerregistry.azurecr.io/<docker_image_name> .
docker push memescontainerregistry.azurecr.io/<docker_image_name>
```

5. **Deploy** Docker containers to Azure Kubernetes Service using `Kubectl` client:
```bash
kubectl apply -f containers-deploy.yaml
```
