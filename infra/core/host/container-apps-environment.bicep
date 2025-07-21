param name string
param location string = resourceGroup().location
param tags object = {}
param logAnalyticsWorkspaceName string = ''

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2021-12-01-preview' = if (!empty(logAnalyticsWorkspaceName)) {
  name: logAnalyticsWorkspaceName
  location: location
  tags: tags
  properties: any({
    retentionInDays: 30
    features: {
      searchVersion: 1
    }
    sku: {
      name: 'PerGB2018'
    }
  })
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2022-03-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: !empty(logAnalyticsWorkspaceName) ? {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    } : null
  }
  dependsOn: [
    logAnalyticsWorkspace
  ]
}

output id string = containerAppsEnvironment.id
output name string = containerAppsEnvironment.name
