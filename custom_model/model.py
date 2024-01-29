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

import kserve
from typing import Dict

from deepsparse import Pipeline


KSERVER_LOGGER_NAME = 'kserver'
DEFAULT_TASK_NAME = 'sentiment-analysis'
DEFAULT_MODEL_PATH = './output'


class NeuralMagicModel(kserve.Model):
    def __init__(self, task: str, model_path: str):
        super().__init__()
        self.name = "neural-magic-model"
        self.task = task
        self.model_path = model_path
        self.load()

    def load(self):
        self.pipeline = Pipeline.create(
            task=self.task,
            model_path=self.model_path)

    def predict(self, request: Dict) -> Dict:
        sequence = request["sequence"]
        result = self.pipeline(sequence)
        return {"predictions": result}


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
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH,
                        help='The path to a model.onnx file, a model folder '
                             'containing the model.onnx')
    args, _ = parser.parse_known_args()

    model = NeuralMagicModel(model_task=args.task, model_path=args.model_path)
    kserve.ModelServer().start([model])