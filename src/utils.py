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
    output_batch, logits_batch = model.inference_for_prompt(prompts=batched_prompts)
    for i, (output, logits) in enumerate(zip(output_batch, logits_batch)):
        num_processed += 1
        question_asked = batched_prompts[i]
        label_mapping = batched_mappings[i]

        result_entry = {
            "task": task,
            "question_number": num_processed,
            "question": question_asked[-1]['content'],
            "answer": output,
            "logits": {
                "A": logits[model.tokenizer.convert_tokens_to_ids("A")],  
                "B": logits[model.tokenizer.convert_tokens_to_ids("B")],  
                "C": logits[model.tokenizer.convert_tokens_to_ids("C")]
            },
            "labels": label_mapping,
            "question_and_answer": f"{question_asked}\n\n{output}",
            "instruction_and_answer": f"{label_mapping}\n\n{output}"
        }
        results.append(result_entry)
        print(result_entry)
        print('\n')
    print(f'processed {num_processed} entries')
    return results, num_processed