import argparse
from typing import List
from finetuning.mistral7B_finetune import Mistral7B_ft
 
from preprocess import *
from utils import *
from evaluators import LogprobsEvaluator
from prompters import FinetunePrompter
import glob
import os

from sklearn.metrics import accuracy_score


def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run evaluate')
    parser.add_argument('--task', default='temporal', type=str, metavar='TASK',
                        help='define the task to evaluate')
    parser.add_argument('--prompt_type', default='zero_shot', type=str,
                        choices=['zero_shot', 'zero_shot_cot'],
                        help='choose prompt type')
    parser.add_argument('--evaluation_type', default=['logprob'], type=str, metavar='EVAL-TYPE', nargs='+',
                        help='choose evaluator type (LLM or logprob)'),
    return parser


def get_task_list_for_eval(model, task, prompt_type):
    """
    Steps to run evaluation on predictions
    - Unzip zero_shot.zip and zero_shot_cot.zip in the predictions folder
    - cd in to zero_shot/ folder and run in terminal: 
            - for file in *.zip; do unzip "$file" -d "${file%.*}"; done
    - do the same for zero_shot_cot/
    """
    task_id = f'{model}_{prompt_type}_{task}'
    pattern = f'predictions/{prompt_type}/{task_id}/{task_id}/{task_id}-*.json'
    print(pattern)
    files = glob.glob(pattern)
    return files

def run_tasks(tasks: List[str], model_name: str, prompt_type: str, evaluation_type: str, ft_model_path:str):
    # only load if llm eval
    if 'llm' in evaluation_type:
        evaluation_model = Mistral7B_ft()
        evaluation_model.get_best_model(ft_model_path)
    
    # Prompter
    evaluation_prompter = FinetunePrompter()
    total_llm_acc: float = 0.
    total_log_prob_acc: float = 0.

    file_paths = get_task_list_for_eval(model_name, tasks, prompt_type)
    print(file_paths)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found. Please first run inference or fix path (check readme or get_task_list_for_eval())")
    else:
        print(f'files that will be evaluated', file_paths)

    for file_path in file_paths:
        print('==============================')
        print('evaluating file', file_path)
        answers = read_json(file_path)


        if 'llm' in evaluation_type:
            evaluator_answers = []

            correct_classes_list = []
            predicted_classes_list = []
            for entry in answers:
                model_answer = entry["answer"]
                true_class  = entry["label"]
                question_asked = entry["question"]

                instruction = evaluation_prompter.create_evaluation_prompt(model_answer=model_answer)
                intermediate_output = evaluation_model.inference_for_prompt(prompt=instruction)
                
                if intermediate_output in ['A', 'B', 'C']:
                    output= entry["label_mapping"][intermediate_output]
                else:
                    output= 'None'
                predicted_classes_list.append(output)
                correct_classes_list.append(true_class)
                evaluator_answers.append((question_asked, output, true_class))
            print("------------------------------------------")
            llm_accuracy = accuracy_score(correct_classes_list, predicted_classes_list)
            print(f'Accuracy for {file_path} and LLM evaluator: {llm_accuracy:.2%}')
            total_llm_acc += llm_accuracy

        if 'logprob' in evaluation_type:
            _, logprob_accuracy, _ = LogprobsEvaluator.compute_logprobs(answers)
            print(f'Accuracy for {file_path} with Logprob evaluator: {logprob_accuracy:.2%}')
            total_log_prob_acc += logprob_accuracy
        
    average_llm_acc = None
    if 'llm' in evaluation_type:
        average_llm_acc =  total_llm_acc / len(file_paths)
    
    average_log_prob_acc = None
    if 'logprob' in evaluation_type:
        average_log_prob_acc =  total_log_prob_acc / len(file_paths)

    return average_llm_acc, average_log_prob_acc

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    if args.prompt_type == 'zero_shot':
        ft_model_path = 'src/finetuning/mistral_atcs_finetune/checkpoint-125'
    elif args.prompt_type == 'zero_shot_cot':
        ft_model_path = 'src/finetuning/mistral_atcs_finetune/checkpoint-275'
    average_llm_acc, average_log_prob_acc = run_tasks(
        args.task, 
        args.model, 
        args.prompt_type,
        args.evaluation_type,
        ft_model_path)
    print('\n\n')
    print('Accuracies for: ', args.model, args.prompt_type, args.task )

    if 'llm' in args.evaluation_type:
        print('Average LLM Accuracy: {:.3f}'.format(average_llm_acc))
    if 'logprob' in args.evaluation_type:
        print('Average Log Probability Accuracy: {:.3f}'.format(average_log_prob_acc))
