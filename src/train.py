from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from pathlib import Path
from swanlab.integration.huggingface import SwanLabCallback
import torch
import json
import pandas as pd


def init_model(model_name: str):
    model_dir = Path(__file__).parent / "model"
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = snapshot_download(model_name, cache_dir = model_dir,)
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",trust_remote_code=True,torch_dtype=torch.bfloat16)
    return model, tokenizer

def main():
    model,tokenizer = init_model("Qwen/Qwen2.5-1.5B-Instruct")
    swanlab_callback = SwanLabCallback(...)

    trainer = Trainer(
        model,
        callbacks=[swanlab_callback],
    )

if __name__ == "__main__":
    main()