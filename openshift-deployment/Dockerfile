FROM python:3.9-slim

COPY ./config.yaml /server-config.yaml

RUN mkdir /.cache && chgrp -R 0 /.cache && chmod -R g=u /.cache

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir deepsparse[transformers,server] && \
    pip3 install --no-cache-dir deepsparse[yolo]
    #pip3 install --no-cache-dir deepsparse-nightly[transformers,yolo,server]

ENTRYPOINT deepsparse.server --config-file /server-config.yaml --port 8080
