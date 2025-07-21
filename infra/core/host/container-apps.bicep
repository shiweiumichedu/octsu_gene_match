param name string
param location string = resourceGroup().location
param tags object = {}

param containerAppsEnvironmentName string
param containerRegistryName string
param logAnalyticsWorkspaceName string = ''

module containerAppsEnvironment 'container-apps-environment.bicep' = {
  name: 'container-apps-environment'
  params: {
    name: containerAppsEnvironmentName
    location: location
    tags: tags
    logAnalyticsWorkspaceName: logAnalyticsWorkspaceName
  }
}

module containerRegistry 'container-registry.bicep' = {
  name: 'container-registry'
  params: {
    name: containerRegistryName
    location: location
    tags: tags
  }
}

output environmentId string = containerAppsEnvironment.outputs.id
output registryName string = containerRegistry.outputs.name
output registryLoginServer string = containerRegistry.outputs.loginServer
