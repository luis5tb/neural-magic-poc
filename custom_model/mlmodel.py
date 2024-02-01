import tarfile

from typing import List

from mlserver import MLModel, types
from mlserver.errors import InferenceError
#from mlserver.utils import get_model_uri

from deepsparse import Pipeline
from sparsezoo import Model


class CustomMLModel(MLModel):
    async def load(self) -> bool:
        self.name = "neural-magic-model"
        self.task = 'sentiment-analysis'
        self.model = 'zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none'
        self._load_model()

        # set ready to signal that model is loaded
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)

        return types.InferenceResponse(
            model_name=self.name,
            outputs=self._predict_outputs(payload),
        )

    def _load_model(self):
        # TODO: load model from file and instantiate class data
        model = Model(self.model, "/models/_mlserver_models")
        model.download()
        self.model_path = model.path + "/deployment"
        deployment_file = model.path + "/deployment.tar.gz"

        untar_directory(deployment_file, model.path)

        self.pipeline = Pipeline.create(
            task=self.task,
            model_path=self.model_path)
        return

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        # TODO: validate request: number of inputs, input tensor names/types, etc.
        #   raise InferenceError on error
        return payload

    def _predict_outputs(self, payload: types.InferenceRequest) -> List[types.ResponseOutput]:
        # get inputs from the request
        inputs = payload.inputs

        # TODO: transform inputs into internal data structures
        sequence = inputs["data"]
        # TODO: send data through the model's prediction logic
        prediction = self.pipeline(sequence)

        # TODO: construct the outputs
        outputs = [
            types.ResponseOutput(
                name='predictions',
                shape=prediction.scores,
                datatype="str",
                data=prediction.labels
            )
        ]

        return outputs
 

def untar_directory(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_path)