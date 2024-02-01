## Adjust the model

Adjust the model as needed, inside custom_model folder

## Build the container for the serving runtime

Build the container with:

`podman build -t quay.io/USER/neural-magic:v1 -f model.Dockerfile .`


And push it to a registry

`podman push quay.io/USER/neural-magic:v1`

## Deploy the serving runtime

Use the `serving_runtime.yaml` to deploy it:

`kubectl apply -f serving_runtime.yaml`

## Create object data store (MinIO) with the model

Create namespace for the object store

`oc new-project object-datastore`

Deploy MinIO:

`oc apply -f minio.yaml`

Download the desired model from sparsezoo

```
from sparsezoo import Model

stub = "zoo:opt-1.3b-opt_pretrain-pruned50_quantW8A8"

model = Model(stub)
model.download()
print(model.path)
```

Copy the files on the Minio bucket `models`


## Create data connection

## Deploy the serving runtime

## Deploy the model



