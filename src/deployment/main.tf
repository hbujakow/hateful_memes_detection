provider "azurerm" {
    features {}
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

resource "azurerm_container_registry" "memescontainerregistry" {
  name                     = "memescontainerregistry"
  resource_group_name      = azurerm_resource_group.memes-resource-group.name
  location                 = azurerm_resource_group.memes-resource-group.location
  sku                      = "Basic"
  admin_enabled            = true
}

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
    container_registry_id = azurerm_container_registry.memescontainerregistry.id

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