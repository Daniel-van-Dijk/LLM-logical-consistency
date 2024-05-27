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

    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        pass


    def _create_question(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        instruction = f'{general_instruction} Premise: "{premise}". Hypothesis: "{hypothesis}". {instruction_format}'
        return [{"role": "user", "content": instruction}]



class ZeroShotPompter(DefaultPrompter):

    def __init__(self):
        print("Using zero shot prompting...")


    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(general_instruction, instruction_format, premise, hypothesis)
    


class ZeroShotPompterCOT(DefaultPrompter):

    def __init__(self):
        print("Using zero shot COT prompting...")
        self.cot_prompt: str = "Let's think step by step."


    def _create_question(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        instruction = f'{general_instruction} Premise: "{premise}". Hypothesis: "{hypothesis}". {instruction_format}'
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": self.cot_prompt}]


    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(general_instruction, instruction_format, premise, hypothesis)



class StarlingZeroShot(ABC):

    def __init__(self):
        print("Zero shot for starling")
    

    def _create_question(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[str]:
        # for starling, we need to remove "Answer: " from the prompt since it's finetuned as shown in "instruction" below.
        instruction_format = instruction_format[:instruction_format.find('?') + 1]
        prompt = f'{general_instruction} Premise: "{premise}". Hypothesis: "{hypothesis}". {instruction_format}'
        instruction = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        return instruction


    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(general_instruction, instruction_format, premise, hypothesis)
    


class StarlingZeroShotCOT(ABC):

    def __init__(self):
        print("Zero shot COT for starling")
        self.cot_prompt: str = "Let's think step by step."
    

    def _create_question(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[str]:
        # for starling, we need to remove "Answer: " from the prompt since it's finetuned as shown in "instruction" below.
        instruction_format = instruction_format[:instruction_format.find('?') + 1]
        prompt = f'{general_instruction} Premise: "{premise}". Hypothesis: "{hypothesis}". {instruction_format}'
        instruction = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {self.cot_prompt}"
        return instruction


    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(general_instruction, instruction_format, premise, hypothesis)



class CollectDataPrompter(DefaultPrompter):

    def __init__(self):
        print("Collecting data for finetuning...")
    

    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        return self._create_question(general_instruction, instruction_format, premise, hypothesis)



class FewShotPrompter(DefaultPrompter):

    def __init__(self):
        self.template_path: str = 'prompts/prompts_few_shot.tsv'
        print("Using few shot prompting...")
        self.prompt_templates = {
            'mcq1': 'Given the premise provided, is the hypothesis: A. entailment or B. neutral or C. contradiction ? \n Answer:',
            'mcq2': 'Given the premise provided, is the hypothesis: A. entailment or B. contradiction or C. neutral ? \n Answer:',
            'mcq3': 'Given the premise provided, is the hypothesis: A. neutral or B. entailment or C. contradiction ? \n Answer:',
            'mcq4': 'Given the premise provided, is the hypothesis: A. neutral or B. contradiction or C. entailment ? \n Answer:',
            'mcq5': 'Given the premise provided, is the hypothesis: A. contradiction or B. neutral or C. entailment ? \n Answer:',
            'mcq6': 'Given the premise provided, is the hypothesis: A. contradiction or B. entailment or C. neutral ? \n Answer:'
        }


    def _get_few_shot_template(self, general_instruction) -> List[Dict[str, str]]:

        template = []
        with open(self.template_path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            # print(reader)
            for _, label_letter, actual_label, premise, hypothesis, instruct_key in reader:
                content = f'{general_instruction} Premise: "{premise}". Hypothesis: "{hypothesis}". {self.prompt_templates[instruct_key]}'
                template.append({"role": "user", "content": content})
                template.append({"role": "assistant", "content": f"{label_letter}"})
                # print('content', content)
                # print('label letter', label_letter)
                # print('actual label: ', actual_label)
                # print()
    
        return template


    def create_instruction(self, general_instruction: str, instruction_format: str, premise: str, hypothesis: str):
        # add few shot prompts
        prompts = self._get_few_shot_template(general_instruction)
        # add the question at hand
        question = self._create_question(general_instruction, instruction_format, premise, hypothesis)
        prompts.extend(question)
        return prompts



class FewShotCOTPrompter(DefaultPrompter):

    def __init__(self):
        self.template_path: str = 'prompts/prompts_few_shot_cot.tsv'
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


class FinetunePrompter(ZeroShotPompter):

    def create_evaluation_prompt(self, model_answer: str) -> List[Dict[str, str]]:
        prompt = f"<s>[INST] Classify the response and give A or B or C or None as an output. {model_answer}.[/INST] "

        return [{"role": "user", "content": prompt}]



class EvaluationPrompter(ZeroShotPompter):

    def create_evaluation_prompt(self, question: str, model_answer: str) -> List[Dict[str, str]]:
        prompt = f"Student Answer: \"{model_answer}\". Did the student pick (a) entailment, (b) neutrality or (c) contradiction? pick one and reply with one word."
        return [{"role": "user", "content": prompt}]



# Class instance creator
def create_prompter_from_str(prompter: str, model: str) -> DefaultPrompter:
    if prompter == "zero_shot":
        if model == "starling7B":
            return StarlingZeroShot()
        return ZeroShotPompter()
        
    elif prompter == "few_shot":
        return FewShotPrompter()
    
    elif prompter == "zero_shot_cot":
        if model == "starling7B":
            return StarlingZeroShotCOT()
        return ZeroShotPompterCOT()
    
    elif prompter == "zero_shot_collect":
        return CollectDataPrompter()
    
    else:
        raise NotImplementedError(f"Unknown prompter: <{prompter}>")



