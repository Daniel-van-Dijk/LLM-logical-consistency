import argparse
from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B

from preprocess import *
from utils import *
from prompts import get_few_shot_template

def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    parser.add_argument('--task', default=['temporal-1'], type=str, metavar='TASK', nargs='+',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-template', default='supported', 
                        choices=['entailment', 'truth', 'supported', 'logically_follow', 'multiple_choice'], 
                        help='choose prompt template')
    parser.add_argument('--prompt_type', default='zero_shot_cot', 
                        choices=['zero_shot', 'zero_shot_cot', 'few_shot', 'few_shot_cot'],
                        help= 'choose prompt type' )
    return parser



def run_tasks(tasks, model_name, prompt_style, prompt_type):
    # TODO: design prompt with CoT
    # TODO: zero-shot vs few-shot
    # TODO: improve templates based on answer variations of model
    prompt_templates = {
        'entailment': 'Does the hypothesis entail or contradict from the premise?',
        'logically_follow': 'Does the hypothesis logically follow from the premise?',
        'truth': 'Given the premise, is the hypothesis true?',
        'supported': 'Is the hypothesis supported by the premise?',
        'multiple_choice': 'Given the premise, is the hypothesis (a) entailment, (b) neutral, or (c) contradiction?'
    }
    instruction_format = prompt_templates[prompt_style]

    if model_name == 'hermes13B':
        model = Hermes13B()
    elif model_name == 'mistral7B':
        model = Mistral7B()
    elif model_name == 'llama3_8B':
        model = LLama3_8B()
    elif model_name == 'starling7B':
        model = Starling7B()

    total_accuracy: float = 0.
    for task in tasks:
        file_path = f'../data/{task}.tsv'
        processed_data = process_tsv(file_path)
        answers = []
        for entry in processed_data:  
            if prompt_type == 'zero_shot_cot':
                instruction_1 = f'Q: Premise- "{entry[1]}" Hypothesis- "{entry[2]}" {instruction_format}? A: Lets think step by step '
                print("Prompt instruction 1: ", instruction_1)
                message_1 = [{"role": "user", "content": instruction_1}]
                output_1 = model.inference_for_prompt(prompt=message_1)
                instruction_2 = instruction_1 + output_1 + 'Therefore the answer among entailment, contradiction and neutral is: '
                print("-------------------------------------------------")
                print("Prompt instruction 2: ", instruction_2)
                message_2 = [{"role": "user", "content": instruction_2}]
                output = model.inference_for_prompt(prompt=message_2)
                print("------------------------------------------------")
                print(f"Model output: {output}")
            if prompt_type == 'few_shot':
                message = get_few_shot_template(instruction=instruction_format)
                output = model.inference_for_prompt(prompt=message)
                print("------------------------------------------------")
                print(f"Model output: {output}")
            else:
                instruction = f'Premise: "{entry[1]}" Hypothesis: "{entry[2]}" {instruction_format}'
                print("Prompt: ", instruction)
                prompt = [{"role": "user", "content": instruction}]
                output = model.inference_for_prompt(prompt=prompt)
                print(f"Model output: {output}")
                
            answers.append((instruction, output, entry[0]))
        
        if prompt_style in ['entailment', 'truth', 'supported', 'logically_follow']:
            # TODO: improve parsing output to evaluate
            results, accuracy, _, _, _ = parse_yes_no_output(answers)
        elif prompt_style == 'multiple_choice':
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
    print(f'Model: {args.model}')
    average_accuracy = run_tasks(args.task, args.model, args.prompt_template, args.prompt_type)
    print('average accuracy: ', average_accuracy)