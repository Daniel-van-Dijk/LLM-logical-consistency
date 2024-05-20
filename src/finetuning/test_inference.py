import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import transformers
from datetime import datetime
from datasets import load_dataset
from peft import PeftModel

import wandb, os
from typing import List, Dict
import torch
from huggingface_hub import login
from mistral7B_finetune import Mistral7B_ft

import argparse


if __name__ == "__main__":


    path="mistral_atcs_finetune/checkpoint-150"

    my_model= Mistral7B_ft()

 
    eval_prompt = "Based on the following' Given the premise provided, is the hypothesis: A. contradiction or B. neutral or C. entailment ? Answer: A. contradiction. Colin being born after Ruth is not in line with the given information that Colin was born before Ruth (based on their birth years).' Give a single letter answer (A or B or C)</s>"

    message_1 = [{"role": "user", "content": eval_prompt}]

    my_model.get_best_model(path)

    output= my_model.inference_for_prompt(message_1)

    print(output)

    print("all good")


    