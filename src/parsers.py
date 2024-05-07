from abc import ABC


class QAParser:

    def __init__(self, instr_style: str) -> str:
        self.instr_style = instr_style


    def create_instruction(self, premise: str, hypothesis: str) -> str:
        return f'Premise: "{premise}" Hypothesis: "{hypothesis}" {self.instr_style}'
    
    def parse_output(self, response: str) -> str:
        pass
    

class MultipleChoiseParser:

    def create_instruction(self, premise: str, hypothesis: str) -> str:
        return f'Premise: "{premise}" Hypothesis: "{hypothesis}". \\
            Given the premise, is the hypothesis (a) entailment, (b) neutral, or (c) contradiction?'
    

    def parse_output(self, response: str) -> str:
        pass
