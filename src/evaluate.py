import argparse

from typing import List

from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B
from models.testLLM import TinyTest

from preprocess import *
from utils import *
from evaluators import RegexEvaluator, LogprobsEvaluator
from prompters import EvaluationPrompter


def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI evaluation with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    parser.add_argument('--task', default=['temporal-1', 'temporal-2'], type=str, metavar='TASK', nargs='+',
                        help='define tasks to evaluate. possible to give multiple')
    parser.add_argument('--prompt-type', default='zero_shot', type=str,
                        choices=['zero_shot', 'zero_shot_cot', 'few_shot', 'few_shot_cot'],
                        help='choose prompt type')
    parser.add_argument('--evaluation-type', default=['regex', 'logprob', 'llm'], type=str, metavar='TASK', nargs='+',
                        help='choose evaluator type'),
    parser.add_argument('--predictions_dir', default='../predictions', type=str, metavar='PREDICTIONS_DIR',
                        help='dir to store data')
    return parser


def run_tasks(tasks: List[str], model_name: str, prompt_type: str, evaluation_type: str, predictions_dir: str):

    # Evaluator is ALWAYS Llama3 
    #evaluation_model = LLama3_8B()
    # Prompters
    evaluation_prompter = EvaluationPrompter()

    total_accuracy: float = 0.

    for task in tasks:
        file_path = f'{predictions_dir}/{model_name}_{prompt_type}_{task}.json'

        exists = check_if_file_exists(file_path)
        
        if not exists:
            raise FileNotFoundError(f"{file_path} doesn't exist. Please first run inference...")
        
        answers = read_json(file_path)

        if 'llm' in evaluation_type:
            evaluator_answers = []
            for entry in answers:
                question_asked, model_answer, true_class = entry
                # ----------------- #
                # -- Second Step -- #
                # ----------------- #
                instruction = evaluation_prompter.create_evaluation_prompt(
                    question=question_asked,
                    model_answer=model_answer
                )
                print("Prompt: ", instruction)
                output = evaluation_model.inference_for_prompt(prompt=instruction)
                print(f"Model output: {output}")

                evaluator_answers.append((question_asked, output, true_class))
            
            # --------------------------------------------- #
            # In Two-Step LLM QA, the answer is always MCQ  #
            # --------------------------------------------- #
            llm_results, llm_accuracy, _, _, _ = RegexEvaluator.parse_multiple_choice(evaluator_answers)
            print(f'Accuracy for {task} and LLM evaluator: {llm_accuracy:.2%}')
            for result in llm_results:
                print(result) 
        
        if 'regex' in evaluation_type:
            regex_results, regex_accuracy, _, _, _ = RegexEvaluator.parse_multiple_choice(answers)
            print(f'Accuracy for {task} and LLM evaluator: {regex_accuracy:.2%}')
            for result in regex_results:
                print(result) 

        if 'logprob' in evaluation_type:
            for answer in answers:
                print(answer)
                exit()
            logprob_results, logprob_accuracy, all_probs = LogprobsEvaluator.compute_logprobs(answers)
            print(f'Accuracy for {task} and Logprob evaluator: {logprob_accuracy:.2%}')
            for result in logprob_results:
                print(result) 
        
        #total_accuracy += accuracy
    
    average_accuracy = total_accuracy / len(tasks)

    return average_accuracy



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    average_accuracy = run_tasks(
        args.task, 
        args.model, 
        args.prompt_type, 
        args.evaluation_type,
        args.predictions_dir
    )
    print('Average accuracy: ', average_accuracy)
