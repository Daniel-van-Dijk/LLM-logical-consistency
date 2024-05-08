from enum import Enum
import csv

class TaskType(Enum):
    BOOLEAN = 'boolean'
    CAUSAL = 'causal'
    COMPARATIVE = 'comparative'
    CONDITIONAL = 'conditional'
    COREFERENCE = 'coreference'
    NEGATION = 'negation'
    QUANTIFIER = 'quantifier'
    RELATIONAL = 'relational'
    SPATIAL = 'spatial'
    TEMPORAL = 'temporal'

    
def get_few_shot_template(template_path: str, instruction: str):
    template = []
    with open(template_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for task, label, premise, hypothesis in reader:
            content = f"{premise} {hypothesis} {instruction}"
            template.append({"role": "user", "content": content})
            template.append({"role": "assistant", "content": f"{label}"})
  
    return template
        