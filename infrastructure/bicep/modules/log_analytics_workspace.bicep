metadata description = 'Creates an Azure Log Analytics Workspace.'

param name string
param location string
param tags object

resource log 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name:  name
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
  }
  tags: tags
}

output logOut string = log.id
