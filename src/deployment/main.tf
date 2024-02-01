provider "azurerm" {
    features {}
}


locals {
  env_variables = {
   DOCKER_REGISTRY_SERVER_URL            = "https://memescontainerregistry.azurecr.io"
   DOCKER_REGISTRY_SERVER_USERNAME       = "" # fill it in
   DOCKER_REGISTRY_SERVER_PASSWORD       = "" # fill it in
   AZURE_CONTAINER_REGISTRY_ID           = "" # fill it in
 }
}

resource "azurerm_resource_group" "memes-resource-group" {
  name     = "memes-resource-group"
  location = "West Europe"
}


resource "azurerm_storage_account" "memes_storage_account" {
  name                     = "sahatefulmemes"
  resource_group_name      = azurerm_resource_group.memes-resource-group.name
  location                 = azurerm_resource_group.memes-resource-group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_log_analytics_workspace" "memes_log_analytics_workspace" {
  name                = "memes-log-analytics-workspace"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "memes_app_environment" {
    name                       = "memes-app-environment"
    location                   = azurerm_resource_group.memes-resource-group.location
    resource_group_name        = azurerm_resource_group.memes-resource-group.name
    log_analytics_workspace_id = azurerm_log_analytics_workspace.memes_log_analytics_workspace.id
}

resource "azurerm_container_app" "hateful_memes_classifier" {
    name                         = "haetful-memes-classifier"
    container_app_environment_id = azurerm_container_app_environment.memes_app_environment.id
    resource_group_name          = azurerm_resource_group.memes-resource-group.name
    revision_mode                = "Single"

    template {
        container {
        name   = "haetful-memes-classifier-app"
        image  = "memescontainerregistry.azurecr.io/hateful_memes_app:latest"
        cpu    = 0.25
        memory = "0.5Gi"
        }
  }
}

resource "azurerm_container_app" "inpainting_api_container" {
    name                         = "inpainting-api"
    container_app_environment_id = azurerm_container_app_environment.memes_app_environment.id
    resource_group_name          = azurerm_resource_group.memes-resource-group.name
    revision_mode                = "Single"

    template {
        container {
        name   = "hateful-memes-inpainting-api-container"
        image  = "memescontainerregistry.azurecr.io/hateful_memes_inpainting_api:latest"
        cpu    = 4
        memory = "8Gi"
        }
  }
}
# Blip2 model probably needs to be deployed in AzureML as well
# resource "azurerm_container_app" "captioning_api_container" { 
#     name                         = "captioning-api"
#     container_app_environment_id = azurerm_container_app_environment.memes_app_environment.id
#     resource_group_name          = azurerm_resource_group.memes-resource-group.name
#     revision_mode                = "Single"

#     template {
#         container {
#         name   = "hateful-memes-captioning-api-container"
#         image  = "memescontainerregistry.azurecr.io/hateful_memes_captioning_api:latest"
#         cpu    = 0.25
#         memory = "0.5Gi"
#         }
#   }
# }

# workspace for RoBERTa (and Blip2 model)

resource "azurerm_key_vault" "memes_key_vault" {
  name                = "memes-key-vault"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
}

resource "azurerm_application_insights" "memes_ai_insights" {
  name                = "memes-ai-insights"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  application_type    = "web"
}

resource "azurerm_machine_learning_workspace" "memes-ml-workspace" {
    name                    = "memes-ml-workspace"
    location                = azurerm_resource_group.memes-resource-group.location
    resource_group_name     = azurerm_resource_group.memes-resource-group.name
    application_insights_id = azurerm_application_insights.memes_ai_insights.id
    storage_account_id      = azurerm_storage_account.memes_storage_account.id
    key_vault_id            = azurerm_key_vault.memes_key_vault.id
    container_registry_id = local.env_variables.AZURE_CONTAINER_REGISTRY_ID

    identity {
        type = "SystemAssigned"
    }
}

resource "azurerm_kubernetes_cluster" "aks_cluster" { #TBC
  name                = "memes-aks-cluster"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  dns_prefix          = "memesAKSDNS"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_DS2_v2"
  }

  # service_principal {
  #   client_id     = "<your-client-id>"
  #   client_secret = "<your-client-secret>"
  # }

  tags = {
    Environment = "Production"
  }
}

output "kube_config" {
  value = azurerm_kubernetes_cluster.aks_cluster.kube_config_raw
}