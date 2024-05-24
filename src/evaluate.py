import argparse

from typing import List

from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B
from models.testLLM import TinyTest
from finetuning.mistral7B_finetune import Mistral7B_ft

from preprocess import *
from utils import *
from evaluators import RegexEvaluator, LogprobsEvaluator
from prompters import EvaluationPrompter, FinetunePrompter
import glob
import os

from sklearn.metrics import accuracy_score


def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    parser.add_argument('--ft_model_path', default="src/finetuning/mistral_atcs_finetune/checkpoint-125", type=str, metavar='PATH',
                        help='model path')
    parser.add_argument('--task', default='temporal', type=str, metavar='TASK',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-type', default='zero_shot', type=str,
                        choices=['zero_shot', 'zero_shot_cot', 'few_shot', 'few_shot_cot'],
                        help='choose prompt type')
    parser.add_argument('--evaluation-type', default=['logprob'], type=str, metavar='TASK', nargs='+',
                        help='choose evaluator type'),
    return parser


def get_task_list_for_eval(model, task, prompt_type):
    """
    put folders for model, task, prompt_type like: starling7B_zero_shot_numerical/ in root directory!
    """
    pattern = f'{model}_{prompt_type}_{task}/{model}_{prompt_type}_{task}-*.json'
    files = glob.glob(pattern)
    return files

def run_tasks(tasks: List[str], model_name: str, prompt_type: str, evaluation_type: str, ft_model_path:str):

    evaluation_model = Mistral7B_ft()
    # Prompters
    evaluation_prompter = FinetunePrompter()
    total_llm_acc: float = 0.
    total_regex_acc: float = 0.
    total_log_prob_acc: float = 0.

    file_paths = get_task_list_for_eval(model_name, tasks, prompt_type)
    print(file_paths)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found. Please first run inference or fix path")
    else:
        print(f'files that will be evaluated', file_paths)
    total_log_prob_acc = 0
    for file_path in file_paths:
        print('==============================')
        print('evaluating file', file_path)

        # exists = check_if_file_exists(file_paths)
        
        # if not exists:
        #     raise FileNotFoundError(f"{file_path} doesn't exist. Please first run inference...")
        
        answers = read_json(file_path)


        if 'llm' in evaluation_type:
            evaluator_answers = []
            evaluation_model.get_best_model(ft_model_path)

            correct_classes_list=[]
            predicted_classes_list=[]
            for entry in answers:
                
                # ----------------- #
                # -- Second Step -- #
                # ----------------- #
                model_answer=entry["answer"]
                true_class=entry["label"]
                question_asked=entry["question"]

                instruction = evaluation_prompter.create_evaluation_prompt(
                    model_answer=model_answer
                )

                #print("------")
                #print("Prompt: ", instruction)
                intermediate_output = evaluation_model.inference_for_prompt(prompt=instruction)
                #print("Intermediate output", intermediate_output)
                if intermediate_output in ['A', 'B', 'C']:
                    output= entry["label_mapping"][intermediate_output]
                else:
                    output= 'None'

                predicted_classes_list.append(output)
                #print(f"Model output: {output},sque true class: {true_class}")
                correct_classes_list.append(true_class)
                evaluator_answers.append((question_asked, output, true_class))
            
            # --------------------------------------------- #
            # In Two-Step LLM QA, the answer is always MCQ  #
            # --------------------------------------------- #
            print("------------------------------------------")
            
         
            llm_accuracy= accuracy_score(correct_classes_list, predicted_classes_list)
            print(f'Accuracy for {file_path} and LLM evaluator: {llm_accuracy:.2%}')

            total_llm_acc += llm_accuracy


        if 'regex' in evaluation_type:
            regex_results, regex_accuracy, _, _, _ = RegexEvaluator.parse_multiple_choice(answers)
            print(f'Accuracy for {file_path} and regex evaluator: {regex_accuracy:.2%}')
            total_regex_acc += regex_accuracy

        if 'logprob' in evaluation_type:
            _, logprob_accuracy, _ = LogprobsEvaluator.compute_logprobs(answers)
            print(f'Accuracy for {file_path} with Logprob evaluator: {logprob_accuracy:.2%}')
            total_log_prob_acc += logprob_accuracy
        
    average_llm_acc = None
    if 'llm' in evaluation_type:
        average_llm_acc =  total_llm_acc / len(file_paths)
    
    average_regex_acc = None
    if 'regex' in evaluation_type:
        average_regex_acc =  total_regex_acc / len(file_paths)
    
    average_log_prob_acc = None
    if 'logprob' in evaluation_type:
        average_log_prob_acc =  total_log_prob_acc / len(file_paths)

    return average_llm_acc, average_regex_acc, average_log_prob_acc


# def run_tasks(tasks: List[str], model_name: str, prompt_type: str, evaluation_type: str):

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    average_llm_acc, average_regex_acc, average_log_prob_acc = run_tasks(
        args.task, 
        args.model, 
        args.prompt_type,
        args.evaluation_type,
        args.ft_model_path)
    print('\n\n')
    print('Accuracies for: ', args.model, args.prompt_type, args.task )

    if 'llm' in args.evaluation_type:
        print('Average LLM Accuracy: {:.3f}'.format(average_llm_acc))
    if 'regex' in args.evaluation_type:
        print('Average Regex Accuracy: {:.3f}'.format(average_regex_acc))
    if 'logprob' in args.evaluation_type:
        print('Average Log Probability Accuracy: {:.3f}'.format(average_log_prob_acc))
