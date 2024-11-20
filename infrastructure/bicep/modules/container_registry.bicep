metadata description = 'Creates an Azure Container Registery.'

param name string
param location string
param tags object

// Container Registry
resource cr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: name
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
  tags: tags
}

output crOut string = cr.id
