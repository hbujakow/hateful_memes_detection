#!/bin/bash

# Variables
resourceGroupName="memes-resource-group"
acrName="memes-conatiner-registry"
imageName="hateful_memes_app"
tag="v1"
location="West Europe"

az group create --name $resourceGroupName --location $location
az acr create --resource-group $resourceGroupName --name $acrName --sku Free
