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

class CollectDataPrompter(ABC):

    def __init__(self):
        print("Collecting data for finetuning...")
    
    def _create_question(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        instruction = f'{general_instruction} Premise: "{premise}". Hypothesis: "{hypothesis}". {instruction_format}'
        return [{"role": "user", "content": instruction}]


    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(general_instruction, instruction_format, premise, hypothesis)


class FewShotPrompter(DefaultPrompter):

    def __init__(self):
        self.template_path: str = '../prompts/prompts_few_shot.tsv'
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
        prompts.extend(question)
        return prompts

class FewShotCOTPrompter(DefaultPrompter):

    def __init__(self):
        self.template_path: str = '../prompts/prompts_few_shot_cot.tsv'
        print("Using few shot prompting with Chain-of-Thought...")


    def _get_few_shot_template(self, instruction: str) -> List[Dict[str, str]]:

        template = []
        with open(self.template_path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for label, premise, hypothesis, instruction_ex, chain_of_thought, answer in reader:
                content = f"{premise} {hypothesis} {instruction_ex}"
                template.append({"role": "user", "content": content})
                template.append({"role": "assistant", "content": f"{chain_of_thought} {answer}"})
    
        return template


    def create_instruction(self, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        # add few shot prompts
        prompts = self._get_few_shot_template(instruction=instruction_format)
        # add the question at hand
        question = self._create_question(instruction_format, premise, hypothesis)
        prompts.extend(question)
        return prompts



class EvaluationPrompter(ZeroShotPompter):

    def create_evaluation_prompt(self, question: str, model_answer: str) -> List[Dict[str, str]]:
        prompt = f"Student Answer: \"{model_answer}\". Did the student pick (a) entailment, (b) neutrality or (c) contradiction? pick one and reply with one word."
        return [{"role": "user", "content": prompt}]


# Class instance creator
def create_prompter_from_str(prompter: str) -> DefaultPrompter:
    if prompter == "zero_shot":
        return ZeroShotPompter()
    elif prompter == "few_shot":
        return FewShotPrompter()
    elif prompter == 'zero_shot_collect':
        return CollectDataPrompter()
    else:
        raise NotImplementedError(f"Unknown prompter: <{prompter}>")