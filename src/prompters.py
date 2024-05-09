from typing import List, Dict
from enum import Enum
from abc import ABC

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



class DefaultPrompter(ABC):

    def create_instruction(self, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        pass


    def _create_question(self, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        instruction = f'Premise: "{premise}" Hypothesis: "{hypothesis}" {instruction_format}'
        return [{"role": "user", "content": instruction}]



class ZeroShotPompter(DefaultPrompter):

    def __init__(self):
        print("Using zero shot prompting...")


    def create_instruction(self, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(instruction_format, premise, hypothesis)



class FewShotPrompter(DefaultPrompter):

    def __init__(self):
        self.template_path: str = './prompts/prompts_few_shot.tsv'
        print("Using few shot prompting...")


    def _get_few_shot_template(self, instruction: str) -> List[Dict[str, str]]:

        template = []
        with open(self.template_path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for task, label, premise, hypothesis in reader:
                content = f"{premise} {hypothesis} {instruction}"
                template.append({"role": "user", "content": content})
                template.append({"role": "assistant", "content": f"{label}"})
    
        return template


    def create_instruction(self, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        # add few shot prompts
        prompts = self._get_few_shot_template(instruction=instruction_format)
        # add the question at hand
        question = self._create_question(instruction_format, premise, hypothesis)
        prompts.append(question)
        return prompts


class EvaluationPrompter(ZeroShotPompter):

    def create_evaluation_prompt(self, question: str, model_answer: str) -> List[Dict[str, str]]:
        prompt = f'Evaluate if this model answer concludes to \\
            (a) entailment, (b) neutrality, (c) contradiction. \\
            {model_answer}'
        return [{"role": "user", "content": prompt}]


# Class instance creator
def create_prompter_from_str(prompter: str) -> DefaultPrompter:
    if prompter == "zero_shot":
        return ZeroShotPompter()
    elif prompter == "few_shot":
        return FewShotPrompter()
    else:
        raise NotImplementedError(f"Unknown prompter: <{prompter}>")