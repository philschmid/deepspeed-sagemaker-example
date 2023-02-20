import os

os.environ["AWS_PROFILE"] = "hf-sm"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace

iam_client = boto3.client("iam")
role = iam_client.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
sess = sagemaker.Session()

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

# hyperparameters, which are passed into the training job
hyperparameters = {
    "training_script": "scripts/run_glue.py",
    "model_name_or_path": "bert-large-uncased",
    "task_name": "sst2",
    "do_train": True,
    "per_device_train_batch_size": 32,
    "num_train_epochs": 3,
    "output_dir": "/opt/ml/model",
    "deepspeed": "configs/ds_z3_offload.json",
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point="ds_launcher.py",
    source_dir=".",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.17",
    pytorch_version="1.10",
    py_version="py38",
    hyperparameters=hyperparameters,
)
# launch training
huggingface_estimator.fit()
