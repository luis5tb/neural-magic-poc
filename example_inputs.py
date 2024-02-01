import json
import numpy as np


data = np.load("sample-model-input.npz")

shape = data["input_ids"].shape
inputs = data["input_ids"].tolist()

print("SHAPE", shape)
print(json.dumps(inputs))

# Example curl command
# curl https://a-neural.apps.ocp4.example.com/v2/models/a/infer \
# -X POST \
# --data '{"inputs" : [{"name" : "X","shape" : [ 128 ],"datatype" : "FP32", "data" : [101, 2339, 2024, 3060, 1011, 4841, 2061, 3376, 1029, 102, 2339, 2024, 6696, 2015, 2061, 3376, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]}'
