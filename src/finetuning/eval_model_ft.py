
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
from sklearn.metrics import accuracy_score
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning processing', add_help=False)
    parser.add_argument('--models_list', default=["mistral", "llama", "starling"], type=str, metavar='models', nargs='+',
                        help='define model names to finetune on. possible to give multiple')
    parser.add_argument('--cot_prompting', default="yes", type=str,
                        help='cot prompts or not?')
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    models_list= args.models_list
    print(f'Models: {models_list}')
    output_name= "_".join(models_list)
    cot_prompting= args.cot_prompting
    print("Do we use cot prompting?:", cot_prompting)

    #model

    

    #dataset
    if cot_prompting=='yes':

        eval_dataset = load_dataset('json', data_files=f'./modeling_data/eval_cot_{output_name}.jsonl', split='train')
        path="mistral_atcs_finetune_cot/checkpoint-275"

    else:
        eval_dataset = load_dataset('json', data_files=f'./modeling_data/eval_{output_name}.jsonl', split='train')
        path="mistral_atcs_finetune/checkpoint-125"

    my_model= Mistral7B_ft()
    my_model.get_best_model(path)
    correct_classes_list=[]
    predicted_classes_list=[]

    for row in eval_dataset:
        if row['output']is not None:
            correct_classes_list.append(row['output'])

            text = f"<s>[INST] Classify the response and give A or B or C or None as an output. {row['input']}.[/INST] "

            message_1 = [{"role": "user", "content": text}]
            output= my_model.inference_for_prompt(message_1)


            predicted_classes_list.append(output)
            print( output, "vs", row['output'])


    print("-------------End of inferences--------------")
    final_score= accuracy_score(correct_classes_list, predicted_classes_list)
    print("Totoal accuracy:", final_score)

    print("Finished")


    