from typing import List

import numpy as np
from mlserver import MLModel, types
from mlserver.errors import InferenceError
from mlserver.utils import get_model_uri
from deepsparse import compile_model

# files with these names are searched for and assigned to model_uri with an
# absolute path (instead of using model URI in the model's settings)
# TODO: set wellknown names to support easier local testing
WELLKNOWN_MODEL_FILENAMES = ["model.json", "model.dat"]


class CustomMLModel(MLModel):
    async def load(self) -> bool:
        # get URI to model data
        model_uri = await get_model_uri(self._settings)
        print("MODEL_URI", model_uri)

        # parse/process file and instantiate the model
        self._load_model_from_file(model_uri)

        # set ready to signal that model is loaded
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=self._predict_outputs(payload),
        )

    def _load_model_from_file(self, file_uri):
        # assume that file_uri is an absolute path
        # TODO: load model from file and instantiate class data
        self.model = compile_model(file_uri, batch_size=128)

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        # TODO: validate request: number of inputs, input tensor names/types, etc.
        #   raise InferenceError on error
        return payload

    def _predict_outputs(self, payload: types.InferenceRequest) -> List[types.ResponseOutput]:
        # get inputs from the request
        inputs = np.array(payload.inputs[0].data).astype(np.uint8)

        # TODO: transform inputs into internal data structures
        # TODO: send data through the model's prediction logic
        outputs = self.model.run([inputs])

        return outputs
