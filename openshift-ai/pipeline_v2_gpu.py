from kfp import dsl
from kfp import compiler
from kfp import kubernetes


@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311',
               packages_to_install=['huggingface-hub','boto3'])
def download_model(model_name: str, destination_path: str,
                   download_option: str):
    if download_option == "HF":
        import subprocess
        print('Starting downloading the model from HF')
        # Execute the huggingface_hub-cli command
        result = subprocess.run(["huggingface-cli", "download", model_name,
                                 "--local-dir", destination_path,
                                 "--local-dir-use-symlinks", "False"],
                                capture_output=True, text=True)
        # Check for errors or output
        if result.returncode == 0:
            print("Model downloaded successfully from HF.")
        else:
            print("Error downloading model:")
            print(result.stderr)

    elif download_option == "S3":
        import os
        import errno
        from boto3 import client

        print('Starting downloading the model from S3')
 
        s3_endpoint_url = os.environ["s3_host"]
        s3_access_key = os.environ["s3_access_key"]
        s3_secret_key = os.environ["s3_secret_access_key"]
        s3_bucket_name = os.environ["s3_bucket"]

        s3_client = client(
            's3', endpoint_url=s3_endpoint_url, aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key, verify=False
        )

        # list all objects in the folder
        objects = s3_client.list_objects(Bucket=s3_bucket_name, Prefix=model_name)

        # download each object in the folder
        for object in objects['Contents']:
            file_name = object['Key']
            local_file_name = os.path.join(destination_path, file_name.replace(model_name, '')[1:])
            if not os.path.exists(os.path.dirname(local_file_name)):
                try:
                    os.makedirs(os.path.dirname(local_file_name))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        print("Error downloading model")
                        raise
            s3_client.download_file(s3_bucket_name, file_name, local_file_name)

        print('Model downloaded successfully from S3.')
    
    elif download_option == "PVC":
        print('Model should be already on the volumen.')
        

@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311',
               packages_to_install=['boto3'])
def upload_model(model_path: str, name: str):
    import os
    from boto3 import client

    print('Starting results upload.')
    s3_endpoint_url = os.environ["s3_host"]
    s3_access_key = os.environ["s3_access_key"]
    s3_secret_key = os.environ["s3_secret_access_key"]
    s3_bucket_name = os.environ["s3_bucket"]

    print(f'Uploading predictions to bucket {s3_bucket_name} '
          f'to S3 storage at {s3_endpoint_url}')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url, aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key, verify=False
    )

    # Walk through the local folder and upload files
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.join(name, local_file_path[len(model_path)+1:])
            s3_client.upload_file(local_file_path, s3_bucket_name, s3_file_path)
            print(f'Uploaded {local_file_path}')

    print('Finished uploading results.')


@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311',
               packages_to_install=["datasets", "auto-gptq==0.7.1", "torch==2.2.1", "sentencepiece"])
def quantize_gpu_model(model_path:str, compress_model_path: str, ds: str):
    # Quantizing an LLM
    from transformers import AutoTokenizer
    from datasets import load_dataset

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    MAX_SEQ_LEN = 512
    NUM_EXAMPLES = 512

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"],
                                                      tokenize=False)}

    print("Loading the dataset and tokenizers")
    dataset = load_dataset(ds, split="train_sft")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds = dataset.shuffle().select(range(NUM_EXAMPLES))
    ds = ds.map(preprocess)

    examples = [
        tokenizer(
            example["text"], padding=False, max_length=MAX_SEQ_LEN,
            truncation=True,
        ) for example in ds
    ]

    print("Loaded the dataset and tokenizers")
    print("Starting the quantization")

    # Apply GPTQ
    quantize_config = BaseQuantizeConfig(
        bits=4,                         # Only support 4 bit
        group_size=128,                 # Set to g=128 or -1 (for channelwise)
        desc_act=False,                 # Marlin does not support act_order=True
        model_file_base_name="model",   # Name of the model.safetensors when we call save_pretrained
    )
    print("Applying GPTQ for quantization")

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config,
        device_map="auto")
    model.quantize(examples)

    gptq_save_dir = f"{model_path}-gptq"
    print(f"Saving gptq model to {gptq_save_dir}")
    model.save_pretrained(gptq_save_dir)
    tokenizer.save_pretrained(gptq_save_dir)

    # Convert to Marlin
    print("Reloading in marlin format")
    marlin_model = AutoGPTQForCausalLM.from_quantized(
        gptq_save_dir,
        use_marlin=True,
        device_map="auto")

    print(f"Saving model in marlin format to {compress_model_path}")
    marlin_model.save_pretrained(compress_model_path)
    tokenizer.save_pretrained(compress_model_path)

    print("Quantization process completed")


@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311',
               packages_to_install=["lm-eval[vllm]", "optimum"])
def eval_model(evaluate: bool, model_path: str, tasks: str, batch_size: str, model_type: str):
    import subprocess
    import os

    if not evaluate:
        return
    model_args = "pretrained=" + model_path  # + ",trust_remote_code=True"

    # Execute the huggingface_hub-cli command
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(["lm_eval",
                             "--model", model_type,
                             "--model_args", model_args,
                             "--tasks", tasks,
                             "--batch_size", batch_size,
                             "--write_out",
                             "--num_fewshot", "0"],
                            capture_output=True, text=True, env=env)

    # Check for errors or output
    if result.returncode == 0:
        print("Model evaluated successfully:")
        print(result.stdout)
    else:
        print("Error evaluating the model:")
        print(result.stderr)
       

@dsl.pipeline
def my_pipeline(model_name: str="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                download_option: str="HF",
                saved_model: str="optimized-gpu-model",
                evaluate:bool=True,
                ):
    """My ML pipeline."""
    BASE_DIR = "/mnt/models/"
    MODEL_DIR = BASE_DIR + "llm"
    COMPRESS_MODEL_DIR = BASE_DIR + "compress-llm"
    # Hardcoded for now
    data_connection = "aws-connection-models"
    volume_name = "models-shared-gpu"
    
    # Create a PVC
    #pvc_cpu = kubernetes.CreatePVC(
    #    # can also use pvc_name instead of pvc_name_suffix to use a pre-existing PVC
    #    pvc_name=shared_volume,
    #    access_modes=['ReadWriteOnce'],
    #    size='20Gi',
    #    storage_class_name='',  # 'standard',
    #)

    # Use the created PVC in the download_model component
    download_task = download_model(
        model_name=model_name,
        download_option=download_option,
        destination_path=MODEL_DIR
    )
    kubernetes.mount_pvc(
        download_task,
        pvc_name=volume_name,
        mount_path=BASE_DIR,  # '/mnt/models',
    )
    kubernetes.use_secret_as_env(download_task,
                                 secret_name=data_connection,
                                 secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                                    'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                                    'AWS_S3_ENDPOINT': 's3_host',
                                                    'AWS_S3_BUCKET':'s3_bucket'})
    
    ds = "HuggingFaceH4/ultrachat_200k"
    quant_llm = quantize_gpu_model(model_path=MODEL_DIR,
                                   compress_model_path=COMPRESS_MODEL_DIR,
                                   ds=ds).after(download_task)
    quant_llm.set_accelerator_limit(1)
    quant_llm.set_accelerator_type('nvidia.com/gpu')

    upload_task = upload_model(model_path=COMPRESS_MODEL_DIR, name=saved_model).after(quant_llm)
    
    eval_base_task = eval_model(evaluate=evaluate,
                                model_path=MODEL_DIR,
                                tasks="hellaswag",
                                batch_size="auto",
                                model_type="hf").after(download_task)
    eval_base_task.set_accelerator_limit(1)
    eval_base_task.set_accelerator_type('nvidia.com/gpu')
    
    eval_quant_task = eval_model(evaluate=evaluate,
                                 model_path=COMPRESS_MODEL_DIR,
                                 tasks="hellaswag",
                                 batch_size="auto",
                                 model_type="vllm").after(quant_llm)
    eval_quant_task.set_accelerator_limit(1)
    eval_quant_task.set_accelerator_type('nvidia.com/gpu')

    ### links   
    kubernetes.mount_pvc(
        quant_llm,
        pvc_name=volume_name,
        mount_path=BASE_DIR,  # '/mnt/models',
    )
    kubernetes.set_timeout(quant_llm, 18000)
    kubernetes.add_toleration(quant_llm,
                              key='nvidia.com/gpu',
                              operator='Exists',
                              effect='NoSchedule')

    kubernetes.use_secret_as_env(upload_task,
                                 secret_name=data_connection,
                                 secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                                    'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                                    'AWS_S3_ENDPOINT': 's3_host',
                                                    'AWS_S3_BUCKET':'s3_bucket'})
    kubernetes.mount_pvc(
        upload_task,
        pvc_name=volume_name,
        mount_path=BASE_DIR,  # '/mnt/models',
    )

    kubernetes.mount_pvc(
        eval_base_task,
        pvc_name=volume_name,
        mount_path=BASE_DIR,  # '/mnt/models',
    )
    kubernetes.set_timeout(eval_base_task, 18000)
    kubernetes.add_toleration(eval_base_task,
                              key='nvidia.com/gpu',
                              operator='Exists',
                              effect='NoSchedule')
    kubernetes.mount_pvc(
        eval_quant_task,
        pvc_name=volume_name,
        mount_path=BASE_DIR,  # '/mnt/models',
    )    
    kubernetes.set_timeout(eval_quant_task, 18000)
    kubernetes.add_toleration(eval_quant_task,
                              key='nvidia.com/gpu',
                              operator='Exists',
                              effect='NoSchedule')


compiler.Compiler().compile(my_pipeline, package_path='pipeline_gpu.yaml')