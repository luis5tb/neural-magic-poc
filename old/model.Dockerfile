FROM ghcr.io/neuralmagic/deepsparse:1.4.2

COPY custom_model custom_model

WORKDIR custom_model
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir /neural_models && chgrp -R 0 /neural_models && chmod -R g=u /neural_models

ENTRYPOINT ["python", "model.py"]


