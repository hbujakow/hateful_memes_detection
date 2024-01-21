# Hateful memes detection application deployment on Azure cloud

**Prerequisites**:
* Docker installed on your local computer.
* Terraform installed on your machine. You can download and install it from [Terraform Downloads](https://www.terraform.io/downloads.html)

1. **Log in** to your Azure acount using Aure client commands:
```bash
az login
az account set --subscription "your-azure-subscription"
```

2. **Create resource group and container registry** using Azure cli commands by executing the `create_azure_registry.sh` script:
```bash
create_azure_registry.sh
```

3. **Build docker image and push the container to Azure registry** using docker commands.
```bash
docker build -t memes-conatiner-registry.azurecr.io/hateful_memes_app ../demo/.
docker push memes-container-registry.azurecr.io/hateful_memes_app
```


4. **Deploy** the infrastructure using Terraform:
```bash
terraform init
terraform plan
terraform apply
```

5. Navigate to the Azure portal and create HTTP triggers in cloud function for each API by inserting the code from respective __init__.py files.

<TODO>
