import csv

from typing import List
from pathlib import Path


def validate_args(args):
    if args.evaluation_type == 'logprobs' and args.prompt_template != 'mcq':
        raise ValueError("Log probability evaluation works only on MCQ prompt inputs.")


def write_tsv(file_path: str, rows: List):
    # store answers for task
    with open(file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        csv_writer.writerows(rows)


def check_if_file_exists(file_path: str) -> bool:

    my_file = Path(file_path)
    if my_file.is_file():
        # file exists
        return True
    
    return False


def process_batch(model, batched_prompts, batched_mappings, task, num_processed, results):
    outputs = model.inference_for_prompt(prompts=batched_prompts)
    for i, output in enumerate(outputs):
        num_processed += 1
        question_asked: str = batched_prompts[i]
        print(question_asked)
        result_entry = {
            "task": task,
            "question_number": num_processed,
            "question": question_asked[-1]['content'],
            "answer": output,
            "labels": batched_mappings[i],
            "question_and_answer": f"{question_asked}\n\n{output}",
            "instruction_and_answer": f"{batched_mappings[i]}\n\n{output}"
        }
        results.append(result_entry)
        print(result_entry)
        print('\n')
    print(f'processed {num_processed} entries')
    return results, num_processed