FROM ghcr.io/neuralmagic/deepsparse:1.4.2

COPY custom_model custom_model

WORKDIR custom_model
RUN pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT ["python", "model.py"]


