metadata description = 'Creates an Azure Key Vault.'

param name string
param location string
param tags object

// Key Vault
resource kv 'Microsoft.KeyVault/vaults@2024-04-01-preview' = {
  name: name
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []
  }
  tags: tags
}

output kvOut string = kv.id
