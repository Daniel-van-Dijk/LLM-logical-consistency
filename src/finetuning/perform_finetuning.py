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


def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning processing', add_help=False)
    parser.add_argument('--models_list', default=["mistral", "llama", "starling"], type=str, metavar='models', nargs='+',
                        help='define model names to finetune on. possible to give multiple')
    parser.add_argument('--cot_prompting', default="yes", type=str,
                        help='cot prompts or not?')

    return parser

# data processing functions
def formater(example):
    text = f"<s>[INST] Classify the response and give A or B or C or None as an output. {example['input']}.[/INST] {example['output']}"

    return text

def prompt_tokens(prompt):
    result = tokenizer(
        formater(prompt),
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    
    return result



if __name__ == "__main__":

    #parser
    args = get_args_parser()
    args = args.parse_args()
    models_list= args.models_list
    print(f'Models: {models_list}')
    output_name= "_".join(models_list)
    cot_prompting= args.cot_prompting
    print("Do we use cot prompting?:", cot_prompting)

    #wandb setup

    wandb.login(key = "add it here")

    if cot_prompting== 'yes' :
        wandb_project = "atcs_finetune_cot"
    else:
        wandb_project = "atcs_finetune"

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    #lora config
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  
        task_type="CAUSAL_LM",
    )

    #model actions
    my_model= Mistral7B_ft()
    model = my_model.model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model = accelerator.prepare_model(model)

    #tokenizer
    tokenizer =my_model.tokenizer

    #data handling
    
    if cot_prompting== 'yes':
        print("yes once again")
        train_dataset = load_dataset('json', data_files=f'./modeling_data/train_cot_{output_name}.jsonl', split='train')
        eval_dataset = load_dataset('json', data_files=f'./modeling_data/eval_cot_{output_name}.jsonl', split='train')
    else:
        train_dataset = load_dataset('json', data_files=f'./modeling_data/train_{output_name}.jsonl', split='train')
        eval_dataset = load_dataset('json', data_files=f'./modeling_data/eval_{output_name}.jsonl', split='train')
    tokenized_train_dataset = train_dataset.map(prompt_tokens)
    tokenized_val_dataset = eval_dataset.map(prompt_tokens)

    #output names 
    base_model_name = "mistral"
    run_name = base_model_name + "_" + wandb_project
    output_dir = "./" + run_name


    #define trainer 
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=500,
            learning_rate=2.5e-5, 
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,             
            logging_dir="./logs",        
            save_strategy="steps",       
            save_steps=25,                
            evaluation_strategy="steps", 
            eval_steps=25,               
            do_eval=True,                
            report_to="wandb",           
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # silence the warnings. re-enable for inference!
    model.config.use_cache = False  

    #perform training
    trainer.train()

