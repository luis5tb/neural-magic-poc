# PIPELINE DEFINITION
# Name: my-pipeline
# Description: My ML pipeline.
# Inputs:
#    download_option: str [Default: 'HF']
#    eval: bool [Default: True]
#    model_name: str [Default: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0']
#    saved_model: str [Default: 'optimized-cpu-model']
#    sparsity_ratio: float [Default: 0.5]
#    sparsity_targets: str [Default: '["re:model.layers.\\d*$"]']
components:
  comp-condition-1:
    dag:
      tasks:
        eval-model:
          cachingOptions:
            enableCache: true
          componentRef:
            name: comp-eval-model
          inputs:
            parameters:
              batch_size:
                runtimeValue:
                  constant: auto
              model_path:
                runtimeValue:
                  constant: /mnt/models/llm
              model_type:
                runtimeValue:
                  constant: hf
              tasks:
                runtimeValue:
                  constant: hellaswag
          taskInfo:
            name: eval-model
        eval-model-2:
          cachingOptions:
            enableCache: true
          componentRef:
            name: comp-eval-model-2
          inputs:
            parameters:
              batch_size:
                runtimeValue:
                  constant: auto
              model_path:
                runtimeValue:
                  constant: /mnt/models/compress-llm
              model_type:
                runtimeValue:
                  constant: sparseml
              tasks:
                runtimeValue:
                  constant: hellaswag
          taskInfo:
            name: eval-model-2
    inputDefinitions:
      parameters:
        pipelinechannel--eval:
          parameterType: BOOLEAN
  comp-download-model:
    executorLabel: exec-download-model
    inputDefinitions:
      parameters:
        destination_path:
          parameterType: STRING
        download_option:
          parameterType: STRING
        model_name:
          parameterType: STRING
  comp-eval-model:
    executorLabel: exec-eval-model
    inputDefinitions:
      parameters:
        batch_size:
          parameterType: STRING
        model_path:
          parameterType: STRING
        model_type:
          parameterType: STRING
        tasks:
          parameterType: STRING
  comp-eval-model-2:
    executorLabel: exec-eval-model-2
    inputDefinitions:
      parameters:
        batch_size:
          parameterType: STRING
        model_path:
          parameterType: STRING
        model_type:
          parameterType: STRING
        tasks:
          parameterType: STRING
  comp-export-model:
    executorLabel: exec-export-model
    inputDefinitions:
      parameters:
        exported_model_path:
          parameterType: STRING
        model_path:
          parameterType: STRING
  comp-sparse-cpu-model:
    executorLabel: exec-sparse-cpu-model
    inputDefinitions:
      parameters:
        compress_model_path:
          parameterType: STRING
        ds:
          parameterType: STRING
        model_path:
          parameterType: STRING
        sparsity_ratio:
          parameterType: NUMBER_DOUBLE
        sparsity_targets:
          parameterType: STRING
  comp-upload-model:
    executorLabel: exec-upload-model
    inputDefinitions:
      parameters:
        model_path:
          parameterType: STRING
        name:
          parameterType: STRING
deploymentSpec:
  executors:
    exec-download-model:
      container:
        args:
        - --executor_input
        - '{{$}}'
        - --function_to_execute
        - download_model
        command:
        - sh
        - -c
        - "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip ||\
          \ python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1\
          \ python3 -m pip install --quiet --no-warn-script-location 'kfp==2.7.0'\
          \ '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"'  &&\
          \  python3 -m pip install --quiet --no-warn-script-location 'huggingface-hub'\
          \ 'boto3' && \"$0\" \"$@\"\n"
        - sh
        - -ec
        - 'program_path=$(mktemp -d)


          printf "%s" "$0" > "$program_path/ephemeral_component.py"

          _KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"

          '
        - "\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import\
          \ *\n\ndef download_model(model_name: str, destination_path: str,\n    \
          \               download_option: str):\n    if download_option == \"HF\"\
          :\n        import subprocess\n        print('Starting downloading the model\
          \ from HF')\n        # Execute the huggingface_hub-cli command\n       \
          \ result = subprocess.run([\"huggingface-cli\", \"download\", model_name,\n\
          \                                 \"--local-dir\", destination_path,\n \
          \                                \"--local-dir-use-symlinks\", \"False\"\
          ],\n                                capture_output=True, text=True)\n  \
          \      # Check for errors or output\n        if result.returncode == 0:\n\
          \            print(\"Model downloaded successfully from HF.\")\n       \
          \ else:\n            print(\"Error downloading model:\")\n            print(result.stderr)\n\
          \n    elif download_option == \"S3\":\n        import os\n        import\
          \ errno\n        from boto3 import client\n\n        print('Starting downloading\
          \ the model from S3')\n\n        s3_endpoint_url = os.environ[\"s3_host\"\
          ]\n        s3_access_key = os.environ[\"s3_access_key\"]\n        s3_secret_key\
          \ = os.environ[\"s3_secret_access_key\"]\n        s3_bucket_name = os.environ[\"\
          s3_bucket\"]\n\n        s3_client = client(\n            's3', endpoint_url=s3_endpoint_url,\
          \ aws_access_key_id=s3_access_key,\n            aws_secret_access_key=s3_secret_key,\
          \ verify=False\n        )\n\n        # list all objects in the folder\n\
          \        objects = s3_client.list_objects(Bucket=s3_bucket_name, Prefix=model_name)\n\
          \n        # download each object in the folder\n        for object in objects['Contents']:\n\
          \            file_name = object['Key']\n            local_file_name = os.path.join(destination_path,\
          \ file_name.replace(model_name, '')[1:])\n            if not os.path.exists(os.path.dirname(local_file_name)):\n\
          \                try:\n                    os.makedirs(os.path.dirname(local_file_name))\n\
          \                except OSError as exc: # Guard against race condition\n\
          \                    if exc.errno != errno.EEXIST:\n                   \
          \     print(\"Error downloading model\")\n                        raise\n\
          \            s3_client.download_file(s3_bucket_name, file_name, local_file_name)\n\
          \n        print('Model downloaded successfully from S3.')\n\n    elif download_option\
          \ == \"PVC\":\n        print('Model should be already on the volumen.')\n\
          \n"
        image: registry.access.redhat.com/ubi9/python-311
    exec-eval-model:
      container:
        args:
        - --executor_input
        - '{{$}}'
        - --function_to_execute
        - eval_model
        command:
        - sh
        - -c
        - "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip ||\
          \ python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1\
          \ python3 -m pip install --quiet --no-warn-script-location 'kfp==2.7.0'\
          \ '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"' && \"\
          $0\" \"$@\"\n"
        - sh
        - -ec
        - 'program_path=$(mktemp -d)


          printf "%s" "$0" > "$program_path/ephemeral_component.py"

          _KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"

          '
        - "\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import\
          \ *\n\ndef eval_model(model_path: str, tasks: str, batch_size: str, model_type:\
          \ str):\n    import subprocess\n    import os\n\n    model_args = \"pretrained=\"\
          \ + model_path  # + \",trust_remote_code=True\"\n\n    # Execute the huggingface_hub-cli\
          \ command\n    env = os.environ.copy()\n    env[\"CUDA_VISIBLE_DEVICES\"\
          ] = \"0\"\n    result = subprocess.run([\"lm_eval\",\n                 \
          \            \"--model\", model_type,\n                             \"--model_args\"\
          , model_args,\n                             \"--tasks\", tasks,\n      \
          \                       \"--batch_size\", batch_size,\n                \
          \             \"--write_out\",\n                             \"--num_fewshot\"\
          , \"0\"],\n                            capture_output=True, text=True, env=env)\n\
          \n    # Check for errors or output\n    if result.returncode == 0:\n   \
          \     print(\"Model evaluated successfully:\")\n        print(result.stdout)\n\
          \    else:\n        print(\"Error evaluating the model:\")\n        print(result.stderr)\n\
          \n"
        image: quay.io/ltomasbo/neural-magic:base_eval-latest
        resources:
          accelerator:
            count: '1'
            type: nvidia.com/gpu
    exec-eval-model-2:
      container:
        args:
        - --executor_input
        - '{{$}}'
        - --function_to_execute
        - eval_model
        command:
        - sh
        - -c
        - "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip ||\
          \ python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1\
          \ python3 -m pip install --quiet --no-warn-script-location 'kfp==2.7.0'\
          \ '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"' && \"\
          $0\" \"$@\"\n"
        - sh
        - -ec
        - 'program_path=$(mktemp -d)


          printf "%s" "$0" > "$program_path/ephemeral_component.py"

          _KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"

          '
        - "\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import\
          \ *\n\ndef eval_model(model_path: str, tasks: str, batch_size: str, model_type:\
          \ str):\n    import subprocess\n    import os\n\n    model_args = \"pretrained=\"\
          \ + model_path  # + \",trust_remote_code=True\"\n\n    # Execute the huggingface_hub-cli\
          \ command\n    env = os.environ.copy()\n    env[\"CUDA_VISIBLE_DEVICES\"\
          ] = \"0\"\n    result = subprocess.run([\"lm_eval\",\n                 \
          \            \"--model\", model_type,\n                             \"--model_args\"\
          , model_args,\n                             \"--tasks\", tasks,\n      \
          \                       \"--batch_size\", batch_size,\n                \
          \             \"--write_out\",\n                             \"--num_fewshot\"\
          , \"0\"],\n                            capture_output=True, text=True, env=env)\n\
          \n    # Check for errors or output\n    if result.returncode == 0:\n   \
          \     print(\"Model evaluated successfully:\")\n        print(result.stdout)\n\
          \    else:\n        print(\"Error evaluating the model:\")\n        print(result.stderr)\n\
          \n"
        image: quay.io/ltomasbo/neural-magic:base_eval-latest
        resources:
          accelerator:
            count: '1'
            type: nvidia.com/gpu
    exec-export-model:
      container:
        args:
        - --executor_input
        - '{{$}}'
        - --function_to_execute
        - export_model
        command:
        - sh
        - -c
        - "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip ||\
          \ python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1\
          \ python3 -m pip install --quiet --no-warn-script-location 'kfp==2.7.0'\
          \ '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"' && \"\
          $0\" \"$@\"\n"
        - sh
        - -ec
        - 'program_path=$(mktemp -d)


          printf "%s" "$0" > "$program_path/ephemeral_component.py"

          _KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"

          '
        - "\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import\
          \ *\n\ndef export_model(model_path: str, exported_model_path: str):\n  \
          \  from sparseml import export\n\n    export(\n        model_path,\n   \
          \     task=\"text-generation\",\n        sequence_length=1024,\n       \
          \ target_path=exported_model_path\n    )\n\n"
        image: quay.io/ltomasbo/neural-magic:sparseml
        resources:
          accelerator:
            count: '1'
            type: nvidia.com/gpu
    exec-sparse-cpu-model:
      container:
        args:
        - --executor_input
        - '{{$}}'
        - --function_to_execute
        - sparse_cpu_model
        command:
        - sh
        - -c
        - "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip ||\
          \ python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1\
          \ python3 -m pip install --quiet --no-warn-script-location 'kfp==2.7.0'\
          \ '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"'  &&\
          \  python3 -m pip install --quiet --no-warn-script-location 'datasets' 'sentencepiece'\
          \ && \"$0\" \"$@\"\n"
        - sh
        - -ec
        - 'program_path=$(mktemp -d)


          printf "%s" "$0" > "$program_path/ephemeral_component.py"

          _KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"

          '
        - "\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import\
          \ *\n\ndef sparse_cpu_model(model_path:str, compress_model_path: str, ds:\
          \ str,\n                     sparsity_ratio: float, sparsity_targets: str):\n\
          \    import sparseml.transformers\n    import torch\n\n    # set the data\
          \ type of the model to bfloat16 and device_map=\"auto\" which\n    # will\
          \ place the model on all the gpus available in the system\n    model = sparseml.transformers.SparseAutoModelForCausalLM.from_pretrained(\n\
          \        model_path,\n        torch_dtype=torch.bfloat16,\n        device_map=\"\
          auto\"\n    )\n\n    recipe = f\"\"\"\n    test_stage:\n      obcq_modifiers:\n\
          \        LogarithmicEqualizationModifier:\n          mappings: [\n     \
          \       [[\"re:.*q_proj\", \"re:.*k_proj\", \"re:.*v_proj\"], \"re:.*input_layernorm\"\
          ],\n            [[\"re:.*gate_proj\", \"re:.*up_proj\"], \"re:.*post_attention_layernorm\"\
          ],\n          ]\n        QuantizationModifier:\n          ignore:\n    \
          \        # These operations don't make sense to quantize\n            -\
          \ LlamaRotaryEmbedding\n            - LlamaRMSNorm\n            - SiLUActivation\n\
          \            - MatMulOutput_QK\n            - MatMulOutput_PV\n        \
          \  post_oneshot_calibration: true\n          scheme_overrides:\n       \
          \     # Enable channelwise quantization for better accuracy\n          \
          \  Linear:\n              weights:\n                num_bits: 8\n      \
          \          symmetric: true\n                strategy: channel\n        \
          \    MatMulLeftInput_QK:\n              input_activations:\n           \
          \     num_bits: 8\n                symmetric: true\n            # For the\
          \ embeddings, only weight-quantization makes sense\n            Embedding:\n\
          \              input_activations: null\n              weights:\n       \
          \         num_bits: 8\n                symmetric: false\n        SparseGPTModifier:\n\
          \          sparsity: {sparsity_ratio}\n          sequential_update: true\n\
          \          quantize: true\n          targets: {sparsity_targets}\n    \"\
          \"\"\n\n    sparseml.transformers.oneshot(\n        model=model,\n     \
          \   dataset=ds,\n        recipe=recipe,\n        output_dir=compress_model_path,\n\
          \    )\n\n"
        image: quay.io/ltomasbo/neural-magic:sparseml
        resources:
          accelerator:
            count: '1'
            type: nvidia.com/gpu
    exec-upload-model:
      container:
        args:
        - --executor_input
        - '{{$}}'
        - --function_to_execute
        - upload_model
        command:
        - sh
        - -c
        - "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip ||\
          \ python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1\
          \ python3 -m pip install --quiet --no-warn-script-location 'kfp==2.7.0'\
          \ '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"'  &&\
          \  python3 -m pip install --quiet --no-warn-script-location 'boto3' && \"\
          $0\" \"$@\"\n"
        - sh
        - -ec
        - 'program_path=$(mktemp -d)


          printf "%s" "$0" > "$program_path/ephemeral_component.py"

          _KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"

          '
        - "\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import\
          \ *\n\ndef upload_model(model_path: str, name: str):\n    import os\n  \
          \  from boto3 import client\n\n    print('Starting results upload.')\n \
          \   s3_endpoint_url = os.environ[\"s3_host\"]\n    s3_access_key = os.environ[\"\
          s3_access_key\"]\n    s3_secret_key = os.environ[\"s3_secret_access_key\"\
          ]\n    s3_bucket_name = os.environ[\"s3_bucket\"]\n\n    print(f'Uploading\
          \ predictions to bucket {s3_bucket_name} '\n          f'to S3 storage at\
          \ {s3_endpoint_url}')\n\n    s3_client = client(\n        's3', endpoint_url=s3_endpoint_url,\
          \ aws_access_key_id=s3_access_key,\n        aws_secret_access_key=s3_secret_key,\
          \ verify=False\n    )\n\n    # Walk through the local folder and upload\
          \ files\n    for root, dirs, files in os.walk(model_path):\n        for\
          \ file in files:\n            local_file_path = os.path.join(root, file)\n\
          \            #s3_file_path = os.path.join(s3_bucket_name, local_file_path[len(model_path)+1:])\n\
          \            s3_file_path = os.path.join(name, local_file_path[len(model_path)+1:])\n\
          \            s3_client.upload_file(local_file_path, s3_bucket_name, s3_file_path)\n\
          \            print(f'Uploaded {local_file_path}')\n\n    print('Finished\
          \ uploading results.')\n\n"
        image: registry.access.redhat.com/ubi9/python-311
pipelineInfo:
  description: My ML pipeline.
  name: my-pipeline
root:
  dag:
    tasks:
      condition-1:
        componentRef:
          name: comp-condition-1
        dependentTasks:
        - download-model
        - sparse-cpu-model
        inputs:
          parameters:
            pipelinechannel--eval:
              componentInputParameter: eval
        taskInfo:
          name: condition-1
        triggerPolicy:
          condition: inputs.parameter_values['pipelinechannel--eval'] == true
      download-model:
        cachingOptions:
          enableCache: true
        componentRef:
          name: comp-download-model
        inputs:
          parameters:
            destination_path:
              runtimeValue:
                constant: /mnt/models/llm
            download_option:
              componentInputParameter: download_option
            model_name:
              componentInputParameter: model_name
        taskInfo:
          name: download-model
      export-model:
        cachingOptions:
          enableCache: true
        componentRef:
          name: comp-export-model
        dependentTasks:
        - sparse-cpu-model
        inputs:
          parameters:
            exported_model_path:
              runtimeValue:
                constant: /mnt/models/exported
            model_path:
              runtimeValue:
                constant: /mnt/models/compress-llm
        taskInfo:
          name: export-model
      sparse-cpu-model:
        cachingOptions:
          enableCache: true
        componentRef:
          name: comp-sparse-cpu-model
        dependentTasks:
        - download-model
        inputs:
          parameters:
            compress_model_path:
              runtimeValue:
                constant: /mnt/models/compress-llm
            ds:
              runtimeValue:
                constant: open_platypus
            model_path:
              runtimeValue:
                constant: /mnt/models/llm
            sparsity_ratio:
              componentInputParameter: sparsity_ratio
            sparsity_targets:
              componentInputParameter: sparsity_targets
        taskInfo:
          name: sparse-cpu-model
      upload-model:
        cachingOptions:
          enableCache: true
        componentRef:
          name: comp-upload-model
        dependentTasks:
        - export-model
        inputs:
          parameters:
            model_path:
              runtimeValue:
                constant: /mnt/models/exported
            name:
              componentInputParameter: saved_model
        taskInfo:
          name: upload-model
  inputDefinitions:
    parameters:
      download_option:
        defaultValue: HF
        isOptional: true
        parameterType: STRING
      eval:
        defaultValue: true
        isOptional: true
        parameterType: BOOLEAN
      model_name:
        defaultValue: TinyLlama/TinyLlama-1.1B-Chat-v1.0
        isOptional: true
        parameterType: STRING
      saved_model:
        defaultValue: optimized-cpu-model
        isOptional: true
        parameterType: STRING
      sparsity_ratio:
        defaultValue: 0.5
        isOptional: true
        parameterType: NUMBER_DOUBLE
      sparsity_targets:
        defaultValue: '["re:model.layers.\\d*$"]'
        isOptional: true
        parameterType: STRING
schemaVersion: 2.1.0
sdkVersion: kfp-2.7.0
---
platforms:
  kubernetes:
    deploymentSpec:
      executors:
        exec-download-model:
          pvcMount:
          - constant: models-shared-cpu
            mountPath: /mnt/models/
          secretAsEnv:
          - keyToEnv:
            - envVar: s3_access_key
              secretKey: AWS_ACCESS_KEY_ID
            - envVar: s3_secret_access_key
              secretKey: AWS_SECRET_ACCESS_KEY
            - envVar: s3_host
              secretKey: AWS_S3_ENDPOINT
            - envVar: s3_bucket
              secretKey: AWS_S3_BUCKET
            secretName: aws-connection-models
        exec-eval-model:
          activeDeadlineSeconds: '18000'
          pvcMount:
          - constant: models-shared-cpu
            mountPath: /mnt/models/
          tolerations:
          - effect: NoSchedule
            key: nvidia.com/gpu
            operator: Exists
        exec-eval-model-2:
          activeDeadlineSeconds: '18000'
          pvcMount:
          - constant: models-shared-cpu
            mountPath: /mnt/models/
          tolerations:
          - effect: NoSchedule
            key: nvidia.com/gpu
            operator: Exists
        exec-export-model:
          activeDeadlineSeconds: '18000'
          pvcMount:
          - constant: models-shared-cpu
            mountPath: /mnt/models/
          tolerations:
          - effect: NoSchedule
            key: nvidia.com/gpu
            operator: Exists
        exec-sparse-cpu-model:
          activeDeadlineSeconds: '18000'
          pvcMount:
          - constant: models-shared-cpu
            mountPath: /mnt/models/
          tolerations:
          - effect: NoSchedule
            key: nvidia.com/gpu
            operator: Exists
        exec-upload-model:
          pvcMount:
          - constant: models-shared-cpu
            mountPath: /mnt/models/
          secretAsEnv:
          - keyToEnv:
            - envVar: s3_access_key
              secretKey: AWS_ACCESS_KEY_ID
            - envVar: s3_secret_access_key
              secretKey: AWS_SECRET_ACCESS_KEY
            - envVar: s3_host
              secretKey: AWS_S3_ENDPOINT
            - envVar: s3_bucket
              secretKey: AWS_S3_BUCKET
            secretName: aws-connection-models
