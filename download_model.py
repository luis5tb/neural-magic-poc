from sparsezoo import Model

stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"

model = Model(stub)
model.download()
print(model.path)
