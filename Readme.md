## Adjust the model

Adjust the model as needed, inside custom_model folder

## Build the container for the serving runtime

Build the container with:

`podman build -t quay.io/ltomasbo/neural-magic:v1 -f model.Dockerfile .`


And push it to a registry

`podman push quay.io/ltomasbo/neural-magic:v1`

## Deploy the serving runtime

Use the `serving_runtime.yaml` to deploy it:

`kubectl apply -f serving_runtime.yaml`

## Create object data store

oc new-project object-datastore

oc apply -f minio.yaml

## Create data connection

## Deploy the serving runtime

## Deploy the model



