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