import csv

from typing import List
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import glob
import os
import numpy as np
import ast
import pandas as pd


def avoid_inf(logit, min_value=-1e10, max_value=1e10):
    # avoid storing of -Infinity and +Infinity
    if logit < min_value:
        return min_value
    if logit > max_value:
        return max_value
    return logit


def process_batch(model, batched_prompts, batched_mappings, batched_labels, task, num_processed, results):
    # from https://discuss.pytorch.org/t/how-to-check-the-gpu-memory-being-used/131220
    device = torch.cuda.get_device_properties(0)
    total_memory = device.total_memory / 1024 / 1024 / 1024  # Convert from bytes to GB
    print(f"Total GPU memory: {total_memory:.2f}GB")
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    processed_batch = False
    attempts = 0
    # keep trying until batch is processed
    while not processed_batch:
        if attempts > 3:
            break
        try:
            # empty cach and torch.no_grad() to increase capacity
            torch.cuda.empty_cache()
            with torch.no_grad():
                output_batch, logits_batch = model.inference_for_prompt(prompts=batched_prompts)
            
            for i, (output, logits, label) in enumerate(zip(output_batch, logits_batch, batched_labels)):
                num_processed += 1

                question_asked = batched_prompts[i]
                label_mapping = batched_mappings[i]
                result_entry = {
                    "task": task,
                    "question_number": num_processed,
                    "question": question_asked,
                    "answer": output,
                    "logits": {
                        "A": avoid_inf(logits[model.tokenizer.convert_tokens_to_ids("A")]),  
                        "B": avoid_inf(logits[model.tokenizer.convert_tokens_to_ids("B")]),  
                        "C": avoid_inf(logits[model.tokenizer.convert_tokens_to_ids("C")])
                    },
                    "logits_mapped": {
                        label_mapping['A']: avoid_inf(logits[model.tokenizer.convert_tokens_to_ids("A")]),
                        label_mapping['B']: avoid_inf(logits[model.tokenizer.convert_tokens_to_ids("B")]),
                        label_mapping['C']: avoid_inf(logits[model.tokenizer.convert_tokens_to_ids("C")])
                    },
                    "label_mapping": label_mapping,
                    "question_and_answer": f"{question_asked}\n\n{output}",
                    "instruction_and_answer": f"{label_mapping}\n\n{output}",
                    "label": label
                }
                results.append(result_entry)

            print(f'processed {num_processed} entries in total')
            processed_batch = True

        except torch.cuda.OutOfMemoryError:
          
            torch.cuda.empty_cache()
            print('out of memory, try again')
            attempts += 1
            print(f'number of attempts {attempts}')
            

    return results, num_processed


def calculate_mismatch_rate(models = ['mistral7B', 'llama3_8B', 'starling7B'], printing=False, cot=False):
    total_mismatch_rate = 0
    for model in models:
        if cot:
            filename = f'mismatch_data/{model}_zero_shot_cot_mismatch.csv'
        else:
            filename = f'mismatch_data/{model}_zero_shot_mismatch.csv'
        df = pd.read_csv(filename)
        # drop text output nans since logits have no nans option
        df = df.dropna(subset=['sentiment'])
        mismatches = 0
        found = 0
        for _, row in df.iterrows():
            # convert str to dict
            logits = ast.literal_eval(row['logits'])
            # strip potential whitespace
            text_answer = row['sentiment'].strip() 
        
            # Check if max value exists
            # for example: "logits": {"A": -10000000000.0, "B": -10000000000.0, "C": -10000000000.0} 
            # Here max() returns A (first value if all equal)
            if logits['A'] == logits['B'] == logits['C']:
                pred_label = 'no_max'
            else:
                pred_label = max(logits, key=logits.get).strip() 

            if pred_label != text_answer:
                mismatches += 1
                if printing:
                    print(f'Mismatch found for: task {row["task"]} question { row["question_number"]} ')
                    print(logits)
                    print("Log prob label:", pred_label)
                    print("Extracted text label:", text_answer)
                    print('Full text answer: ', row['answer'])
                    print("\n")
          
        mismatch_rate = mismatches / len(df)
        total_mismatch_rate += mismatch_rate
        if cot:
            print(f'Mismatch rate for {model} with CoT: {mismatch_rate * 100:.2f}%')
        else:
            print(f'Mismatch rate for {model}: {mismatch_rate * 100:.2f}%')

    print(f'Average mismatch rate of the models {total_mismatch_rate / len(models) * 100:.2f} %')
    

def validate_args(args):
    if args.evaluation_type == 'logprobs' and args.prompt_template != 'mcq':
        raise ValueError("Log probability evaluation works only on MCQ prompt inputs.")


def write_tsv(file_path: str, rows: List):
    # store answers for task
    with open(file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        csv_writer.writerows(rows)

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def check_if_file_exists(file_path: str) -> bool:

    my_file = Path(file_path)
    if my_file.is_file():
        # file exists
        return True
    
    return False

def get_task_list(task_name):   
    files = glob.glob(f'data/{task_name}-*.tsv')
    file_names = [os.path.basename(file)[:-4] for file in files]
    return file_names
