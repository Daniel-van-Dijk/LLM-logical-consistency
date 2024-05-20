import csv

from typing import List
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import glob
import os

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
                        "A": logits[model.tokenizer.convert_tokens_to_ids("A")],  
                        "B": logits[model.tokenizer.convert_tokens_to_ids("B")],  
                        "C": logits[model.tokenizer.convert_tokens_to_ids("C")]
                    },
                    "logits_mapped": {
                        label_mapping['A']: logits[model.tokenizer.convert_tokens_to_ids("A")],
                        label_mapping['B']: logits[model.tokenizer.convert_tokens_to_ids("B")],
                        label_mapping['C']: logits[model.tokenizer.convert_tokens_to_ids("C")]
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


