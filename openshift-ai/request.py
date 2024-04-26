import requests
import json

import gradio as gr
from openai import OpenAI


#URL = "https://SERVING_RUNTIME-predictor-NAMESPACE.apps.devcluster.openshift.com"
#URL = "http://SERVER:8000/v1"

MODEL_AUX = "/mnt/models-aux"
MODEL = "/mnt/models"
MODEL_VAR = "/var/models"


def get_answer(question, url):
    client = OpenAI(base_url=url, api_key="EMPTY")
    model = client.models.list().data[0][1]
    #model = client.models.list().data[0].id
    print(f"Accessing model API '{model}'")

    #completion = client.completions.create(model=model, prompt=question, max_tokens=100, temperature=0.2)
    # Completion API
    stream = False
    prompt = question
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1000,
        temperature=0.2,
        n=1,
        stream=stream)

    if stream:
        for c in completion:
            print(c)
    else:
        print(f"\n----- Prompt:\n{prompt}")
        print(f"\n----- Completion:\n{completion.choices[0].text}")

    return completion.choices[0].text


def get_answer_chat(question, url):
    client = OpenAI(base_url=url, api_key="EMPTY")
    model = client.models.list().data[0][1]
    print(f"Accessing model API '{model}'")

    msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=msg,
        stream=True,
        max_tokens=1000,
        temperature=0.2)
    #return completion.choices[0].message.content
    response_content = []
    for chunk in completion:
        chunk_message = chunk.choices[0].delta
        if chunk_message.content:
            response_content.append(chunk_message.content)
    return ''.join(response_content)


def get_answer_req(question, url, model):
    headers = {"Content-Type": "application/json"}
    url = url + "/v1/chat/completions"

    data = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Use less than 300 words. " + question}
            #{"role": "user", "content": question}
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


def get_answer_req2(question, url, model):
    headers = {"Content-Type": "application/json"}
    url = url + "/v1/completions"


    data = {
        "model": model,
        "max_tokens": 100,
        "temperature": 0,
        "prompt": question,
        }
    resp = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

    print(resp.text)

    if resp.status_code == 200:
        response_text = resp.text
        data = json.loads(response_text)
        actual_response = data["choices"][0]['text']
        return actual_response
    else:
        print("Error:", resp.status_code, resp.text)
        return None

##iface = gr.Interface(fn=get_answer,
#iface = gr.Interface(fn=get_answer_chat,
#                     inputs=["text", gr.Dropdown(choices=[URL])],
#                     outputs="text")

iface = gr.Interface(fn=get_answer_req,
#iface = gr.Interface(fn=get_answer_req2,
#                     inputs=["text", gr.Dropdown(choices=[URL]), gr.Dropdown(choices=[MODEL])],
                     inputs=["text", gr.Dropdown(choices=[URL]), gr.Dropdown(choices=[MODEL, MODEL_AUX, MODEL_VAR])],
                     outputs="text")

iface.launch()
