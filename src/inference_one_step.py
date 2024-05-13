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
    parser.add_argument('--prompt-template', default='mcq1', type=str,
                        choices=['mcq_letters', 'mcq_words'], 
                        help='choose prompt template')
    parser.add_argument('--prompt-type', default='zero_shot', type=str,
                        choices=['zero_shot', 'zero_shot_cot', 'few_shot', 'few_shot_cot'],
                        help='choose prompt type')
    parser.add_argument('--evaluation_type', default='regex', type=str, choices=['regex', 'logprobs'])
    return parser


def run_tasks(tasks: List[str], model_name: str, prompt_style: str, prompt_type: str, evaluation_type: str):
    # TODO: REMOVE non MCQ
    prompt_templates = {
        'mcq1': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Given the premise provided, is the hypothesis: A. entailment, B. neutral or C. contradiction ? \n Answer: ',
        'mcq2': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Given the premise provided, is the hypothesis: A. entailment, B. contradiction or C. neutral ? \n Answer: ',
        'mcq3': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Given the premise provided, is the hypothesis: A. neutral, B. entailment or C. contradiction ? \n Answer: ',
        'mcq4': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Given the premise provided, is the hypothesis: A. neutral, B. contradiction or C. entailment ? \n Answer: ',
        'mcq5': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Given the premise provided, is the hypothesis: A. contradiction, B. neutral or C. entailment ? \n Answer: ',
        'mcq6': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Given the premise provided, is the hypothesis: A. contradiction, B. entailment or C. neutral ? \n Answer: '
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

    total_accuracy: float = 0.

    prompter: DefaultPrompter = create_prompter_from_str(prompt_type)

    for task in tasks:
        file_path = f'../data/{task}.tsv'
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
            output, generated_dict = model.inference_for_prompt(prompt=instruction)
            print(f"Model output: {output}")
            
            question_asked: str = instruction[-1]["content"]
            answers.append((question_asked, output, generated_dict, entry[0]))
        
        # TODO: ADD log prob evaluation!!!!!!
        if evaluation_type == 'logprobs':
            if prompt_style == 'mcq1' :
                results, accuracy, all_probs = compute_logprobs(answers)
            else:
                print("Define parse function for other prompt templates")
                exit()
        else: 
            if prompt_style == 'mcq1':
                # TODO: improve parsing output to evaluate
                # results, accuracy, _, _, _ = parse_yes_no_output(answers)
            # elif prompt_style == 'mcq_letters':
                results, accuracy, _, _, _ = parse_multiple_choice(answers)
            else:
                print("Define parse function for other prompt templates")
                exit()
        
        print(f'Accuracy for {task}: {accuracy:.2%}')
        for result in results:
            print(result) 
        total_accuracy += accuracy
    
    average_accuracy = total_accuracy / len(tasks)

    return average_accuracy


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    validate_args(args)
    print(f'Model: {args.model}')
    average_accuracy = run_tasks(args.task, args.model, args.prompt_template, args.prompt_type, args.evaluation_type)
    print('Average accuracy: ', average_accuracy)