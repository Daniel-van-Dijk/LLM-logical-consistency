import argparse

from typing import List
import random
from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B

from preprocess import *
from utils import *
from prompters import create_prompter_from_str, CollectDataPrompter


def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    # comparative, spatial, temporal, quantifier, numerical
    parser.add_argument('--task', default=['temporal-1'], type=str, metavar='TASK', nargs='+',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-template', default='supported', type=str,
                        choices=['mcq'], 
                        help='choose prompt template')
    parser.add_argument('--prompt-type', default='zero_shot_collect', type=str,
                        choices=['zero_shot_collect'],
                        help='choose prompt type')
    parser.add_argument('--evaluation_type', default='None', type=str, choices=['None'])
    return parser


def run_tasks(tasks: List[str], model_name: str, prompt_style: str, prompt_type: str):

    general_instruction = 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter.'
    prompt_templates = {
        'mcq1': 'Given the premise provided, is the hypothesis: A. entailment, B. neutral or C. contradiction ? \n Answer: ',
        'mcq2': 'Given the premise provided, is the hypothesis: A. entailment, B. contradiction or C. neutral ? \n Answer: ',
        'mcq3': 'Given the premise provided, is the hypothesis: A. neutral, B. entailment or C. contradiction ? \n Answer: ',
        'mcq4': 'Given the premise provided, is the hypothesis: A. neutral, B. contradiction or C. entailment ? \n Answer: ',
        'mcq5': 'Given the premise provided, is the hypothesis: A. contradiction, B. neutral or C. entailment ? \n Answer: ',
        'mcq6': 'Given the premise provided, is the hypothesis: A. contradiction, B. entailment or C. neutral ? \n Answer: '
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

    for task in tasks:
        print('\n\n\n')
        print('==========================================')
        print(f'Collecting 50 responses for task: {task}' )
        file_path = f'../data/{task}.tsv'
        processed_data = process_tsv(file_path)
        answers = []
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
            answers.append((question_asked, output, entry[0]))
    return None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    validate_args(args)
    task_list = ['comparative-1', 'comparative-3', 'spatial-1', 'spatial-9', 
                 'temporal-1', 'temporal-27', 'quantifier-1', 'quantifier-13', 
                 'numerical-1', 'numerical-14']

    print(f'Collecting responses for model: {args.model}')
    average_accuracy = run_tasks(args.task, args.model, args.prompt_template, args.prompt_type)
    print('Average accuracy: ', average_accuracy)