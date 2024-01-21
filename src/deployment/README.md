# Hateful Memes Detection application deployment on Azure

The deployment of the system is done on the Azure cloud by utilizing Azure Cloud Function and Azure Web App services for hosting the APIs and demo web page.

**Prerequisites**:
* Docker installed on your local computer.
* Terraform installed on your machine. You can download and install it from [Terraform Downloads](https://www.terraform.io/downloads.html).

1. **Log in** to your Azure acount using Aure client commands:
```bash
az login
az account set --subscription "your-azure-subscription"
```

2. **Create resource group and container registry** using Azure client commands by executing the `create_azure_registry.sh` script:
```bash
create_azure_registry.sh
```

3. **Build Docker image and push the container to Azure Container Registry** using Docker commands.
```bash
cd ../demo/ # navigate to demo folder
docker build -t memes-conatiner-registry.azurecr.io/hateful_memes_app .
docker push memes-container-registry.azurecr.io/hateful_memes_app
```


4. **Deploy** the infrastructure using Terraform:
```bash
terraform init
terraform plan
terraform apply
```
5. **Navigate** to project root folder and **create** HTTP triggers in cloud function for each API by execute the following Azure client commmands:
```bash
cd <path/to/root/directory> # modify this line of code by inserting appropriate path
az functionapp function update --name captioning_api --resource-group memes_rg --function-name inpainting_api --code inpainting/cloud_function --runtime python --handler __init__.main --authlevel anonymous
az functionapp function update --name captioning_api --resource-group memes_rg --function-name captioning_api --code captions/cloud_function --runtime python --handler __init__.main --authlevel anonymous
az functionapp function update --name captioning_api --resource-group memes_rg --function-name classification_api --code procap/cloud_function --runtime python --handler __init__.main --authlevel anonymous
```
