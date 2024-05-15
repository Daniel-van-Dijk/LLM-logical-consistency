import argparse

from typing import List

from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from models.llama3_8B import LLama3_8B
from models.starling7B import Starling7B

from preprocess import *
from utils import *
from evaluators import RegexEvaluator
from prompters import EvaluationPrompter


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
    parser.add_argument('--evaluation-type', default='regex', type=str,
                        choices=['regex', 'logprob', 'llm'],
                        help='choose evaluator type')
    return parser


def run_tasks(tasks: List[str], model_name: str, prompt_style: str, prompt_type: str, evaluation_type: str):

    # Evaluator is ALWAYS Llama3 
    evaluation_model = LLama3_8B()

    # Prompters
    evaluation_prompter = EvaluationPrompter()

    total_accuracy: float = 0.

    for task in tasks:
        file_path = f'../predictions/{model_name}_{prompt_style}_{prompt_type}_{task}.tsv'

        exists = check_if_file_exists(file_path)
        
        if not exists:
            raise FileNotFoundError(f"{file_path} doesn't exist. Please first run inference...")
        
        answers = process_tsv(file_path)

        if evaluation_type == 'llm':
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
            results, accuracy, _, _, _ = RegexEvaluator.parse_multiple_choice(evaluator_answers)
        
        elif evaluation_type == 'regex':
            results, accuracy, _, _, _ = RegexEvaluator.parse_multiple_choice(answers)

        elif evaluation_type == 'logprob':
            # TODO: waiting for this implementation
            pass
        else:
            raise NotImplementedError(f"Evaluation type: {evaluation_type} not implemented...")
        
        print(f'Accuracy for {task} and {evaluation_type}: {accuracy:.2%}')
        for result in results:
            print(result) 
        
        total_accuracy += accuracy
    
    average_accuracy = total_accuracy / len(tasks)

    return average_accuracy



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    average_accuracy = run_tasks(
        args.task, 
        args.model, 
        args.prompt_template, 
        args.prompt_type, 
        args.evaluation_type
    )
    print('Average accuracy: ', average_accuracy)
