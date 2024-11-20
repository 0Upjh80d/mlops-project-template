metadata description = 'Creates an Azure Machine Learning Workspace.'

param name string
param location string
param stoacctid string
param kvid string
param appinsightid string
param crid string
param tags object

// Optional encryption parameters (if required)
param keyIdentifier string = ''
param keyVaultArmId string = ''


// AML Workspace
resource amls 'Microsoft.MachineLearningServices/workspaces@2024-07-01-preview' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    tier: 'Basic'
    name: 'basic'
  }
  properties: {
    storageAccount: stoacctid
    keyVault: kvid
    applicationInsights: appinsightid
    containerRegistry: crid
    encryption: union(
      {
        status: keyIdentifier != '' && keyVaultArmId != '' ? 'Enabled' : 'Disabled'
      },
      keyIdentifier != '' && keyVaultArmId != '' ? {
        keyVaultProperties: {
          keyIdentifier: keyIdentifier
          keyVaultArmId: keyVaultArmId
        }
      } : {}
    )
  }
  tags: tags
}

output amlsName string = amls.name
