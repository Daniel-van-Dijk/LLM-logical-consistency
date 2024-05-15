import argparse

from typing import List

from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B

from preprocess import *
from utils import *
from prompters import create_prompter_from_str, DefaultPrompter


def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    parser.add_argument('--task', default=['temporal-1'], type=str, metavar='TASK', nargs='+',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-template', default='supported', type=str,
                        choices=['entailment', 'truth', 'supported', 'logically_follow', 'mcq'], 
                        help='choose prompt template')
    parser.add_argument('--prompt-type', default='zero_shot', type=str,
                        choices=['zero_shot', 'zero_shot_cot', 'few_shot', 'few_shot_cot'],
                        help='choose prompt type')
    return parser


def run_tasks(tasks: List[str], model_name: str, prompt_style: str, prompt_type: str) -> List:
    # TODO: REMOVE non MCQ
    prompt_templates = {
        'entailment': 'Does the hypothesis entail or contradict from the premise?',
        'logically_follow': 'Does the hypothesis logically follow from the premise?',
        'truth': 'Given the premise, is the hypothesis true?',
        'supported': 'Is the hypothesis supported by the premise?',
        'mcq': 'Given the premise, is the hypothesis (a) entailment, (b) neutral, or (c) contradiction?'
    }
    instruction_format: str = prompt_templates[prompt_style]

    if model_name == 'hermes13B':
        model = Hermes13B()
    elif model_name == 'mistral7B':
        model = Mistral7B()
    elif model_name == 'llama3_8B':
        model = LLama3_8B()
    elif model_name == 'starling7B':
        model = Starling7B()

    prompter: DefaultPrompter = create_prompter_from_str(prompt_type)

    for task in tasks:
        file_path = f'../data/{task}.tsv'
        file_output_path = f'../predictions/{model_name}_{prompt_style}_{prompt_type}_{task}.tsv'

        processed_data = process_tsv(file_path)
        answers = []
        for entry in processed_data:
            # ---------------- #
            # -- First Step -- #
            # ---------------- #
            instruction = prompter.create_instruction(
                premise=entry[1], 
                hypothesis=entry[2], 
                instruction_format=instruction_format
            )
            print("Prompt: ", instruction)
            output = model.inference_for_prompt(prompt=instruction)

            # post process
            output = output.replace('\n', ' ').replace('\r', '')
            print(f"Model output: {output}")
            
            question_asked: str = instruction[-1]["content"]
            answers.append((question_asked, output, entry[0]))
        
        # write answers in tsv file
        write_tsv(file_output_path, answers)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    average_accuracy = run_tasks(
        args.task, 
        args.model, 
        args.prompt_template, 
        args.prompt_type
    )
    print('Average accuracy: ', average_accuracy)