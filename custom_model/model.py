# Copyright 2024 Red Hat, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import tarfile

import kserve
from typing import Dict

from deepsparse import Pipeline
from sparsezoo import Model


KSERVER_LOGGER_NAME = 'kserver'
DEFAULT_TASK_NAME = 'sentiment-analysis'
DEFAULT_ZOO_MODEL_NAME = 'zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none'


class NeuralMagicModel(kserve.Model):
    def __init__(self, task: str, zoo_model: str):
        self.name = "neural-magic-model"
        super().__init__(self.name)
        self.task = task
        self.model = zoo_model
        self.load()

    def load(self):
        model = Model(self.model, "/neural_models")
        model.download()
        self.model_path = model.path + "/deployment"
        deployment_file = model.path + "/deployment.tar.gz"

        untar_directory(deployment_file, model.path)

        self.pipeline = Pipeline.create(
            task=self.task,
            model_path=self.model_path)

    def predict(self, request: Dict) -> Dict:
        sequence = request["sequence"]
        result = self.pipeline(sequence)
        return {"predictions": result}


def untar_directory(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument('--task', default=DEFAULT_TASK_NAME,
                        help='[custom|question_answering|qa|'
                             'text_classification|glue|sentiment_analysis|'
                             'token_classification|ner|'
                             'zero_shot_text_classification|'
                             'transformers_embedding_extraction|'
                             'image_classification|yolo|yolov8|yolact|'
                             'information_retrieval_haystack|haystack|'
                             'embedding_extraction|open_pif_paf|'
                             'text_generation|opt|bloom|chatbot|chat|'
                             'code_generation|codegen]')
    parser.add_argument('--zoo-model', default=DEFAULT_ZOO_MODEL_NAME)
    args, _ = parser.parse_known_args()

    model = NeuralMagicModel(task=args.task, zoo_model=args.zoo_model)
    kserve.ModelServer().start([model])