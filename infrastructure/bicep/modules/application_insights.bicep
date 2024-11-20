metadata description = 'Creates an Azure Application Insight.'

param name string
param location string
param workspaceResourceId string
param tags object

// Application Insights
resource appinsight 'Microsoft.Insights/components@2020-02-02' = {
  name: name
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspaceResourceId
  }
  tags: tags
}

output appinsightOut string = appinsight.id
