import csv

from typing import List
from pathlib import Path
import torch
import torch.nn.functional as F
import json

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


def process_batch(model, batched_prompts, batched_mappings, batched_labels, task, num_processed, results):
    with torch.no_grad():
        output_batch, logits_batch = model.inference_for_prompt(prompts=batched_prompts)
    for i, (output, logits, label) in enumerate(zip(output_batch, logits_batch, batched_labels)):
        num_processed += 1
        question_asked = batched_prompts[i]
        label_mapping = batched_mappings[i]
        result_entry = {
            "task": task,
            "question_number": num_processed,
            "question": question_asked[-1]['content'],
            "answer": output,
            # logits: logits of the letters
            "logits": {
                "A": logits[model.tokenizer.convert_tokens_to_ids("A")],  
                "B": logits[model.tokenizer.convert_tokens_to_ids("B")],  
                "C": logits[model.tokenizer.convert_tokens_to_ids("C")]
            },
            # logits_mapped: saves the logits of labels directly by using label_mapping
            "logits_mapped": {
                label_mapping['A'] : logits[model.tokenizer.convert_tokens_to_ids("A")],
                label_mapping['B'] : logits[model.tokenizer.convert_tokens_to_ids("B")],
                label_mapping['C'] : logits[model.tokenizer.convert_tokens_to_ids("C")]
            },
            # label_mapping: mapping between labels and letters of this question
            "label_mapping": label_mapping,
            "question_and_answer": f"{question_asked}\n\n{output}",
            "instruction_and_answer": f"{label_mapping}\n\n{output}",
            # ground truth label of this question
            "label" : label
        }
        results.append(result_entry)
    print(f'processed {num_processed} entries in total')
    return results, num_processed