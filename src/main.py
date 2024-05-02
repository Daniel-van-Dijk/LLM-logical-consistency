import argparse
from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B

from preprocess import *
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    parser.add_argument('--task', default=['temporal-1'], type=str, metavar='TASK', nargs='+',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-template', default='supported', choices=['entailment', 'truth', 'supported', 'logically_follow'], help='choose prompt template')
    return parser



def run_tasks(tasks, model_name, prompt_style):
    # TODO: design prompt with CoT
    # TODO: zero-shot vs few-shot
    # TODO: improve templates based on answer variations of model
    prompt_templates = {
        'entailment': 'Does the hypothesis entail or contradict from the premise?',
        'logically_follow': 'Does the hypothesis logically follow from the premise?',
        'truth': 'Given the premise, is the hypothesis true?',
        'supported': 'Is the hypothesis supported by the premise?'
    }
    instruction_format = prompt_templates[prompt_style]

    if model_name == 'hermes13B':
        model = Hermes13B()
    elif model_name == 'mistral7B':
        model = Mistral7B()
    elif model_name == 'llama3_8B':
        model = LLama3_8B()

    for task in tasks:
        file_path = f'../data/{task}.tsv'
        processed_data = process_tsv(file_path)
        answers = []
        for entry in processed_data:  
            instruction = f'Premise: "{entry[1]}" Hypothesis: "{entry[2]}" {instruction_format}'
            print("Prompt: ", instruction)
            prompt = [{"role": "user", "content": instruction}]
            output = model.inference_for_prompt(prompt=prompt)
            print(f"Model output: {output}")
            
            answers.append((instruction, output, entry[0]))
        
        if prompt_style in ['entailment', 'truth',  'supported', 'logically_follow']:
            # TODO: improve parsing output to evaluate
            results, accuracy, _, _, _ = parse_yes_no_output(answers)
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
    average_accuracy = run_tasks(args.task, args.model, args.prompt_template)
    print('average accuracy: ', average_accuracy)