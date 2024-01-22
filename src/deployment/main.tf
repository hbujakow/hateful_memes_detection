provider "azurerm" {
    features {}
}


locals {
  function_app_settings = {
    PROCAP_API_URL = azurerm_linux_function_app.classification_api.default_hostname
    CAPTION_API_URL = azurerm_linux_function_app.captioning_api.default_hostname
    INPAINT_API_URL = azurerm_linux_function_app.inpainting_api.default_hostname
  }
  env_variables = {
   DOCKER_REGISTRY_SERVER_URL            = "https://memescontainerregistry.azurecr.io"
   DOCKER_REGISTRY_SERVER_USERNAME       = "memescontainerregistry"
   DOCKER_REGISTRY_SERVER_PASSWORD       = "PTJS8WnKSV1IjXeoO94c9E6Xz7TCTD6o0kHhQYFa5m+ACRDmH5BG"
 }
}

resource "azurerm_resource_group" "memes-resource-group" {
  name     = "memes-resource-group"
  location = "West Europe"
}

resource "azurerm_service_plan" "memes_aps" {
  name                = "memes-appservice-plan"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  sku_name            = "Y1"
  os_type             = "Linux"
}

resource "azurerm_linux_web_app" "memes_app_service" {
  name                  = "memes-app-service"
  location              = azurerm_resource_group.memes-resource-group.location
  resource_group_name   = azurerm_resource_group.memes-resource-group.name
  service_plan_id       = azurerm_service_plan.memes_aps.id
  app_settings          = local.function_app_settings

  site_config {
    # always_on           = true
    # linux_fx_version    = "DOCKER|memescontainerregistry.azurecr.io/hateful_memes_app:latest"
  }
}

resource "azurerm_linux_function_app" "captioning_api" {
  name                = "memes-captioning-function-app"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  # service_plan_id = azurerm_service_plan.api_memes_aps.id
  service_plan_id = azurerm_service_plan.memes_aps.id

  storage_account_name = azurerm_storage_account.memes_storage_account.name
  storage_account_access_key = azurerm_storage_account.memes_storage_account.primary_access_key
  site_config {
  }
}

resource "azurerm_linux_function_app" "inpainting_api" {
  name                = "memes-inpainting-function-app"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  service_plan_id = azurerm_service_plan.memes_aps.id

  storage_account_name = azurerm_storage_account.memes_storage_account.name
  storage_account_access_key = azurerm_storage_account.memes_storage_account.primary_access_key
  site_config {
  }
}

resource "azurerm_linux_function_app" "classification_api" {
  name                = "memes-classification-function-app"
  location            = azurerm_resource_group.memes-resource-group.location
  resource_group_name = azurerm_resource_group.memes-resource-group.name
  service_plan_id = azurerm_service_plan.memes_aps.id

  storage_account_name = azurerm_storage_account.memes_storage_account.name
  storage_account_access_key = azurerm_storage_account.memes_storage_account.primary_access_key
  site_config {
  }
}

resource "azurerm_storage_account" "memes_storage_account" {
  name                     = "sahatefulmemes"
  resource_group_name      = azurerm_resource_group.memes-resource-group.name
  location                 = azurerm_resource_group.memes-resource-group.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}