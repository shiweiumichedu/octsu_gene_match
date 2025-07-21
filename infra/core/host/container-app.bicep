param name string
param location string = resourceGroup().location
param tags object = {}

param containerAppsEnvironmentId string
param containerRegistryName string
param containerName string = 'main'
param exists bool = false
param external bool = true
param targetPort int = 80
param containerCpuCoreCount string = '0.5'
param containerMemory string = '1.0Gi'
param env array = []

resource existingApp 'Microsoft.App/containerApps@2022-03-01' existing = if (exists) {
  name: name
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2022-02-01-preview' existing = {
  name: containerRegistryName
}

resource app 'Microsoft.App/containerApps@2022-03-01' = {
  name: name
  location: location
  tags: union(tags, {'azd-service-name': containerName})
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      activeRevisionsMode: 'single'
      ingress: external ? {
        external: external
        targetPort: targetPort
        transport: 'auto'
      } : null
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          image: exists ? existingApp.properties.template.containers[0].image : 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          name: containerName
          env: env
          resources: {
            cpu: json(containerCpuCoreCount)
            memory: containerMemory
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
      }
    }
  }
}

output identityPrincipalId string = app.identity.principalId
output name string = app.name
output uri string = external ? 'https://${app.properties.configuration.ingress.fqdn}' : ''
output imageName string = split(app.properties.template.containers[0].image, ':')[0]
