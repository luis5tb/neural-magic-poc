import tarfile

import numpy as np
from typing import List

from mlserver import MLModel, types
from mlserver.utils import get_model_uri

from deepsparse import Engine


class CustomMLModel(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        print("MODEL_URI", model_uri)

        self._load_model_from_file(model_uri)

        # set ready to signal that model is loaded
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)

        return types.InferenceResponse(
            model_name=self.name,
            outputs=self._predict_outputs(payload),
        )

    def _load_model_from_file(self, file_uri):
        self.engine = Engine(file_uri)

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        # TODO: validate request: number of inputs, input tensor names/types, etc.
        #   raise InferenceError on error
        return payload

    def _predict_outputs(self, payload: types.InferenceRequest) -> List[types.ResponseOutput]:
        # get inputs from the request
        inputs = np.array(payload.inputs[0].data).astype(np.uint8)

        # TODO: transform inputs into internal data structures
        # TODO: send data through the model's prediction logic
        outputs = self.engine.run([inputs])

        # TODO: construct the outputs
        #outputs = [
        #    types.ResponseOutput(
        #        name='predictions',
        #        shape=prediction.scores,
        #        datatype="str",
        #        data=prediction.labels
        #    )
        #]

        return outputs
