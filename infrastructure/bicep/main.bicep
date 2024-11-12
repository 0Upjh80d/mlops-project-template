targetScope = 'subscription'

param location string = 'eastus'
param prefix string
param postfix string
param env string

param tags object = {
  Owner: 'mlops-v2'
  Project: 'mlops-v2'
  Environment: env
  Toolkit: 'bicep'
  Name: prefix
}

var baseName = !empty(postfix) ? '${prefix}-${postfix}-${env}' : '${prefix}-${env}'
var resourceGroupName = 'rg-${baseName}'
var abbrs = loadJsonContent('abbreviations.json')

resource rg 'Microsoft.Resources/resourceGroups@2024-07-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

// Storage Account
module st './modules/storage_account.bicep' = {
  name: 'st'
  scope: resourceGroup(rg.name)
  params: {
    name: '${abbrs.storageStorageAccounts}${prefix}${postfix}${env}'
    location: location
    tags: tags
  }
}

// Key Vault
module kv './modules/key_vault.bicep' = {
  name: 'kv'
  scope: resourceGroup(rg.name)
  params: {
    name: '${abbrs.keyVaultVaults}${baseName}'
    location: location
    tags: tags
  }
}

// Log Analytics Workspace
module log './modules/log_analytics_workspace.bicep' = {
  name: 'log'
  scope: resourceGroup(rg.name)
  params: {
    name: '${abbrs.operationalInsightsWorkspaces}${baseName}'
    location: location
    tags: tags
  }
}

// App Insights
module appi './modules/application_insights.bicep' = {
  name: 'appi'
  scope: resourceGroup(rg.name)
  params: {
    name: '${abbrs.insightsComponents}${baseName}'
    location: location
    workspaceResourceId: log.outputs.logOut
    tags: tags
  }
}

// Container Registry
module cr './modules/container_registry.bicep' = {
  name: 'cr'
  scope: resourceGroup(rg.name)
  params: {
    name: '${abbrs.containerRegistryRegistries}${prefix}${postfix}${env}'
    location: location
    tags: tags
  }
}

// AML workspace
module mlw './modules/aml_workspace.bicep' = {
  name: 'mlw'
  scope: resourceGroup(rg.name)
  params: {
    name: '${abbrs.machineLearningServicesWorkspaces}${baseName}'
    location: location
    stoacctid: st.outputs.stoacctOut
    kvid: kv.outputs.kvOut
    appinsightid: appi.outputs.appinsightOut
    crid: cr.outputs.crOut
    tags: tags
  }
}

// AML compute cluster
module mlwcc './modules/aml_computecluster.bicep' = {
  name: 'mlwcc'
  scope: resourceGroup(rg.name)
  params: {
    location: location
    workspaceName: mlw.outputs.amlsName
  }
}
