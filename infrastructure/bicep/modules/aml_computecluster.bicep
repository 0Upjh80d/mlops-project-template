metadata description = 'Creates an Azure Machine Learning Compute Cluster.'

param location string
param computeClusterName string = 'cpu-cluster'
param workspaceName string

resource amlci 'Microsoft.MachineLearningServices/workspaces/computes@2024-07-01-preview' = {
  name: '${workspaceName}/${computeClusterName}'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      subnet: null
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
      }
    }
  }
}
