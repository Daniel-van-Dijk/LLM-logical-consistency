import argparse
from models.hermes13B import Hermes13B
from models.mistral7B import Mistral7B
from preprocess import *

def get_args_parser():
    parser = argparse.ArgumentParser('LoNLI inference with LLMs', add_help=False)
    parser.add_argument('--model', default='mistral7B', type=str, metavar='MODEL',
                        help='model to run inference on')
    return parser


def main(args):
    if args.model == 'hermes13B':
        model = Hermes13B()
    elif args.model == 'mistral7B':
        model = Mistral7B()

    file_path = '../data/temporal-1.tsv'
    processed_data = process_tsv(file_path)
    for entry in processed_data[:10]:  
        # TODO: design prompt with CoT
        # TODO: zero-shot vs few-shot
        instruction = f'Premise: "{entry[1]}" Hypothesis: "{entry[2]}" Does the hypothesis logically follow from the premise? Think step by step'
        print("Prompt: ", instruction)
        prompt = [{"role": "user", "content": instruction}]
        output = model.inference_for_prompt(prompt=prompt)
        print(f"Model output: {output}")
        # TODO: parse output to evaluate

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(f'Model: {args.model}')
    main(args)