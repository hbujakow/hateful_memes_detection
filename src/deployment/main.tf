provider "azurerm" {
    features {}
}

resource "azurerm_resource_group" "memes_rg" {
  name     = "memes-resource-group"
  location = "West Europe"
}

resource "azurerm_app_service_plan" "memes_aps" {
  name                = "streamlit-appservice-plan"
  location            = azurerm_resource_group.memes_rg.location
  resource_group_name = azurerm_resource_group.memes_rg.name
  kind                = "Linux"

  sku {
    tier = "Free"
    size = "F1"
  }
}

resource "azurerm_app_service" "memes_app_service" {
  name                = "streamlit-appservice"
  location            = azurerm_resource_group.memes_rg.location
  resource_group_name = azurerm_resource_group.memes_rg.name
  app_service_plan_id = azurerm_app_service_plan.memes_ap_service.id

  site_config {
    linux_fx_version = "DOCKER|memescontainerregistry/hateful_memes_app:latest"
  }

  app_settings = {
    "WEBSITES_PORT" = "8501"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_function_app" "classification_api" {
  name                = "memes-classification-function-app"
  location            = azurerm_resource_group.memes_rg.location
  resource_group_name = azurerm_resource_group.memes_rg.name
  app_service_plan_id = azurerm_app_service_plan.memes_aps.id

  storage_account_name = azurerm_storage_account.memes_storage_account.name
  storage_account_access_key = azurerm_storage_account.memes_storage_account.primary_access_key

  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME" = "python"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_function_app" "captioning_api" {
  name                = "memes-captioning-function-app"
  location            = azurerm_resource_group.memes_rg.location
  resource_group_name = azurerm_resource_group.memes_rg.name
  app_service_plan_id = azurerm_app_service_plan.memes_aps.id

  storage_account_name = azurerm_storage_account.memes_storage_account.name
  storage_account_access_key = azurerm_storage_account.memes_storage_account.primary_access_key

  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME" = "python"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_function_app" "inpainting_api" {
  name                = "memes-inpainting-function-app"
  location            = azurerm_resource_group.memes_rg.location
  resource_group_name = azurerm_resource_group.memes_rg.name
  app_service_plan_id = azurerm_app_service_plan.memes_aps.id

  storage_account_name = azurerm_storage_account.memes_storage_account.name
  storage_account_access_key = azurerm_storage_account.memes_storage_account.primary_access_key

  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME" = "python"
  }

  identity {
    type = "SystemAssigned"
  }
}


resource "azurerm_storage_account" "memes_storage_account" {
  name                     = "memes-storage-account"
  resource_group_name      = azurerm_resource_group.memes_rg.name
  location                 = azurerm_resource_group.memes_rg.location
  account_tier             = "Free"
  account_replication_type = "LRS"
}


locals {
  function_app_settings = {
    PROCAP_API_URL = azurerm_function_app.classification_api.default_hostname
    CAPTION_API_URL = azurerm_function_app.captioning_api.default_hostname
    INPAINT_API_URL = azurerm_function_app.inpainting_ai.default_hostname
  }
}

resource "azurerm_function_app_settings" "settings" {
  name                = "example-settings"
  resource_group_name = azurerm_resource_group.memes_rg.name
  app_name            = azurerm_app_service.memes_app_service.name
  settings            = local.function_app_settings
}
