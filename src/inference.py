import argparse

from typing import List
import random
from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B
from models.testLLM import TinyTest
import os
from preprocess import *
from utils import *
from prompters import *
import time
import json
from datetime import datetime



def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI inference', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    parser.add_argument('--task', default=['temporal-1'], type=str, metavar='TASK', nargs='+',
                        help='define taqsks to evaluate. possible to give multiple')
    parser.add_argument('--run_all', default=None, type=str, metavar='TASK Type',
                        help='define tasks to evaluate')
    parser.add_argument('--prompt-type', default='zero_shot', type=str,
                        choices=['zero_shot', 'zero_shot_cot', 'few_shot', 'few_shot_cot'],
                        help='choose prompt type')
    parser.add_argument('--output_dir', default='../predictions', type=str, metavar='OUTPUT_DIR',
                        help='dir to store data')
    parser.add_argument('--batch_size', default=24, type=int, metavar='BATCH_SIZE',
                        help='batch size for inference')
    return parser




def run_tasks(tasks: List[str], model_name: str, prompt_type: str, batch_size: int, output_dir: str) -> List:
    print("Run started at", datetime.now())
    start_time = time.time()
    general_instruction = 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter.'
    prompt_templates = {
        'mcq1': 'Given the premise provided, is the hypothesis: A. entailment or B. neutral or C. contradiction ? \n Answer:',
        'mcq2': 'Given the premise provided, is the hypothesis: A. entailment or B. contradiction or C. neutral ? \n Answer:',
        'mcq3': 'Given the premise provided, is the hypothesis: A. neutral or B. entailment or C. contradiction ? \n Answer:',
        'mcq4': 'Given the premise provided, is the hypothesis: A. neutral or B. contradiction or C. entailment ? \n Answer:',
        'mcq5': 'Given the premise provided, is the hypothesis: A. contradiction or B. neutral or C. entailment ? \n Answer:',
        'mcq6': 'Given the premise provided, is the hypothesis: A. contradiction or B. entailment or C. neutral ? \n Answer:'
    }
    # keep correspondence of labels since we shuffle answer options 
    label_mappings = {
        'mcq1': {'A': 'entailment', 'B': 'neutral', 'C': 'contradiction'},
        'mcq2': {'A': 'entailment', 'B': 'contradiction', 'C': 'neutral'},
        'mcq3': {'A': 'neutral', 'B': 'entailment', 'C': 'contradiction'},
        'mcq4': {'A': 'neutral', 'B': 'contradiction', 'C': 'entailment'},
        'mcq5': {'A': 'contradiction', 'B': 'neutral', 'C': 'entailment'},
        'mcq6': {'A': 'contradiction', 'B': 'entailment', 'C': 'neutral'}
    }
    prompter: DefaultPrompter = create_prompter_from_str(prompt_type)

    if model_name == 'hermes13B':
        model = Hermes13B()
    elif model_name == 'mistral7B':
        model = Mistral7B()
    elif model_name == 'llama3_8B':
        model = LLama3_8B()
    elif model_name == 'starling7B':
        model = Starling7B()
        prompter = StarlingZeroShot()
    elif model_name == 'tinytest':
        model = TinyTest()

    os.makedirs(output_dir, exist_ok=True)
    for task in tasks:
        print(f"{task} started at {datetime.now()}")
        results = []
        print('\n\n\n')
        print('==========================================')
        print(f'Collecting predictions for task: {task}')
        file_path = f'../data/{task}.tsv'
        processed_data = process_tsv(file_path)

        batched_prompts, batched_mappings, batched_labels = [], [], []
        num_processed = 0
        for entry in processed_data:
            # Pick random shuffle of answer options to avoid selection bias and store corresponding labels
            random_template_key = random.choice(list(prompt_templates.keys()))
            instruction_format = prompt_templates[random_template_key]
            label_mapping = label_mappings[random_template_key]

            instruction = prompter.create_instruction(
                general_instruction=general_instruction,
                premise=entry[1],
                hypothesis=entry[2],
                instruction_format=instruction_format
            )
            batched_prompts.append(instruction)
            batched_mappings.append(label_mapping)
            batched_labels.append(entry[0])

            # Fill batch with randomly shuffled prompts
            if len(batched_prompts) == batch_size:
                results, num_processed = process_batch(model, batched_prompts, batched_mappings, batched_labels, task, num_processed, results)
                # empty batch again
                batched_prompts, batched_mappings = [], []

        # process potential remaining samples which did not fit in the final batch
        if batched_prompts:
            results, num_processed = process_batch(model, batched_prompts, batched_mappings, batched_labels, task, num_processed, results)

        print(len(results))

        output_path = f'{output_dir}/{model_name}_{prompt_type}_{task}.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Total time taken: {elapsed_time} minutes")
    return None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    if args.run_all is None:
        task_list = args.task
    else:
        task_list = get_task_list(args.run_all)
    
    print("Task list: ", task_list)
    average_accuracy = run_tasks(
        args.task, 
        args.model, 
        args.prompt_type,
        args.batch_size,
        args.output_dir
    )
    print('Average accuracy: ', average_accuracy)
