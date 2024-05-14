import argparse

from typing import List
import random
import json
import os
from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B
import time

from preprocess import *
from utils import *
from prompters import create_prompter_from_str, CollectDataPrompter


def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    
    task_list = ['comparative-1', 'comparative-3', 'spatial-1', 'spatial-9', 
                 'temporal-1', 'temporal-27', 'quantifier-1', 'quantifier-13', 
                 'numerical-1', 'numerical-14']
    parser.add_argument('--task', default=task_list, type=str, metavar='TASK', nargs='+',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-template', default='supported', type=str,
                        choices=['mcq'], 
                        help='choose prompt template')
    parser.add_argument('--prompt-type', default='zero_shot_collect', type=str,
                        choices=['zero_shot_collect'],
                        help='choose prompt type')
    parser.add_argument('--evaluation_type', default='None', type=str, choices=['None'])
    parser.add_argument('--output_dir', default='finetune_data', type=str, metavar='OUTPUT_DIR',
                        help='dir to store data')
    return parser


def run_tasks(tasks: List[str], model_name: str, output_dir: str):
    start_time = time.time()
    general_instruction = 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter.'
    prompt_templates = {
        'mcq1': 'Given the premise provided, is the hypothesis: A. entailment or B. neutral or C. contradiction ? \n Answer: ',
        'mcq2': 'Given the premise provided, is the hypothesis: A. entailment or B. contradiction or C. neutral ? \n Answer: ',
        'mcq3': 'Given the premise provided, is the hypothesis: A. neutral or B. entailment or C. contradiction ? \n Answer: ',
        'mcq4': 'Given the premise provided, is the hypothesis: A. neutral or B. contradiction or C. entailment ? \n Answer: ',
        'mcq5': 'Given the premise provided, is the hypothesis: A. contradiction or B. neutral or C. entailment ? \n Answer: ',
        'mcq6': 'Given the premise provided, is the hypothesis: A. contradiction or B. entailment or C. neutral ? \n Answer: '
    }

    if model_name == 'hermes13B':
        model = Hermes13B()
    elif model_name == 'mistral7B':
        model = Mistral7B()
    elif model_name == 'llama3_8B':
        model = LLama3_8B()
    elif model_name == 'starling7B':
        model = Starling7B()

    collect_prompter = CollectDataPrompter()

    """
    Logical tasks are: comparative, spatial, temporal, quantifier, numerical
    So if we take 100 per task (50 from 2 separate files), we have 500 samples per model.
    """

    results = []

    for task in tasks:
        print('\n\n\n')
        print('==========================================')
        print(f'Collecting 50 responses for task: {task}' )
        file_path = f'../data/{task}.tsv'
        processed_data = process_tsv(file_path)
        for num, entry in enumerate(processed_data[:50]):
            # pick random shuffle of answer options to avoid selection bias
            random_template_key = random.choice(list(prompt_templates.keys()))
            instruction_format = prompt_templates[random_template_key]

            instruction = collect_prompter.create_instruction(
                general_instruction=general_instruction,
                premise=entry[1], 
                hypothesis=entry[2], 
                instruction_format=instruction_format
            )
            
            print(f'Question number: {num}')
            print("Prompt: ", instruction)
            output = model.inference_for_prompt(prompt=instruction)
            print(f"Model output: {output}")
            print('==========================================')
            print('\n')
            
            question_asked: str = instruction[-1]["content"]
            result_entry = {
            "task": task,
            "question_number": num,
            "question": question_asked,
            "answer": output,
            "question_and_answer": f"{question_asked}\n\n{output}",
            "instruction_and_answer": f"{instruction_format}\n\n{output}"
            }
            results.append(result_entry)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'{model_name}_finetune_data.json')
    with open(output_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_file_path}")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Total time taken: {elapsed_time} minutes")
    return None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    validate_args(args)
    print(f'Collecting responses for model: {args.model}')
    average_accuracy = run_tasks(args.task, args.model, args.output_dir)
    print('Average accuracy: ', average_accuracy)