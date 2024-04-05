import requests
import json

import gradio as gr
from openai import OpenAI


URL = "https://neural-magic-llm-poc-predictor-neural-magic-poc.apps.devcluster.openshift.com"


MODEL = "hf:neuralmagic/TinyLlama-1.1B-Chat-v0.4-pruned50-quant-ds"
MODEL_DEFAULT = "/mnt/models/deployment"
MODEL_AUX = "/mnt/models-aux"

def get_answer(question, url, model):
    client = OpenAI(base_url=url, api_key="EMPTY")
    model = client.models.list().data[0][1]
    print(f"Accessing model API '{model}'")

    completion = client.completions.create(model=model, prompt=question, max_tokens=100, temperature=0.2)
    return completion.choices[0].text

def get_answer_req(question, url, model):
    headers = {"Content-Type": "application/json"}
    #url = url + "/v1/completions"
    url = url + "/v1/chat/completions"

    data = {
        "model": model,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]}
    resp = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

    print(resp.text)

    if resp.status_code == 200:
        response_text = resp.text
        data = json.loads(response_text)
        actual_response = data["choices"][0]['message']['content']
        return actual_response
    else:
        print("Error:", resp.status_code, resp.text)
        return None

#iface = gr.Interface(fn=get_answer,
#                     inputs=["text", gr.Dropdown(choices=[URL])],
##                     inputs=["text", gr.Dropdown(choices=[URL])],
#                     outputs="text")

iface = gr.Interface(fn=get_answer_req,
#                     inputs=["text", gr.Dropdown(choices=[URL]), gr.Dropdown(choices=[MODEL])],             
                     inputs=["text", gr.Dropdown(choices=[URL]), gr.Dropdown(choices=[MODEL, MODEL_AUX, MODEL_DEFAULT])],
                     outputs="text")

iface.launch()