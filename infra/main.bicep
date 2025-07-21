targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment that can be used as part of naming resource convention')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Id of the user or app to assign application roles')
param principalId string

// Optional parameters to override the default azd resource naming conventions.
// Add the following to main.parameters.json to set the resource name:
//     {
//       "resourceGroupName": {
//         "value": "myGroupName"
//       }
//     }
@description('Name of the resource group. Generated if not provided.')
param resourceGroupName string = ''

@description('Name of the container apps environment. Generated if not provided.')
param containerAppsEnvironmentName string = ''

@description('Name of the container registry. Generated if not provided.')
param containerRegistryName string = ''

@description('Name of the log analytics workspace. Generated if not provided.')
param logAnalyticsName string = ''

@description('Name of the container app for web service. Generated if not provided.')
param webContainerAppName string = ''

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var tags = { 'azd-env-name': environmentName }

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

// Container apps host (including container registry)
module containerApps './core/host/container-apps.bicep' = {
  name: 'container-apps'
  scope: rg
  params: {
    name: 'app'
    containerAppsEnvironmentName: !empty(containerAppsEnvironmentName) ? containerAppsEnvironmentName : '${abbrs.appManagedEnvironments}${resourceToken}'
    containerRegistryName: !empty(containerRegistryName) ? containerRegistryName : '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    logAnalyticsWorkspaceName: logAnalyticsName
  }
}

// Web frontend
module web './core/host/container-app.bicep' = {
  name: 'web'
  scope: rg
  params: {
    name: !empty(webContainerAppName) ? webContainerAppName : '${abbrs.appContainerApps}web-${resourceToken}'
    location: location
    containerAppsEnvironmentId: containerApps.outputs.environmentId
    containerRegistryName: containerApps.outputs.registryName
    exists: false
    containerCpuCoreCount: '1.0'
    containerMemory: '2.0Gi'
    containerName: 'web'
    env: []
    external: true
    targetPort: 8000
  }
}

// App outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_RESOURCE_GROUP string = rg.name

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerApps.outputs.registryLoginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerApps.outputs.registryName

output SERVICE_WEB_IDENTITY_PRINCIPAL_ID string = web.outputs.identityPrincipalId
output SERVICE_WEB_NAME string = web.outputs.name
output SERVICE_WEB_URI string = web.outputs.uri
output SERVICE_WEB_IMAGE_NAME string = web.outputs.imageName
