import re

from typing import List
import torch
import numpy as np

class RegexEvaluator:
    @staticmethod
    def parse_multiple_choice(answers_labels: List):
        """ 
        Function where answer is expected to be answerable with a multiple choice style
        Expected one of: (a) entailment, (b) neutral, (c) contradiction
        """
        correct, total = 0, 0
        contradictions, entailments, neutrals, nans = 0, 0, 0, 0
        results = []

        for prompt, answer, actual_label in answers_labels:

            answer_lower = answer.lower().strip()
            is_entailment = bool(re.search(r'A.', answer_lower)) or \
                            bool(re.search(r'entailment', answer_lower))
            is_contradiction = bool(re.search(r'\(c\)', answer_lower)) or \
                            bool(re.search(r'contradiction', answer_lower))
            is_neutral = bool(re.search(r'\(b\)', answer_lower)) or \
                            bool(re.search(r'neutral', answer_lower))

            if is_entailment:
                transformed_label = 'entailment'
                entailments += 1
            elif is_contradiction:
                transformed_label = 'contradiction'
                contradictions += 1
            elif is_neutral:
                transformed_label = 'neutral'
                neutrals += 1
            else:
                nans += 1

            total += 1
            if transformed_label == actual_label:
                correct += 1 

            results.append((prompt, transformed_label, actual_label))
        accuracy = correct / total
        print('\n')
        print("Accuracy:", accuracy)
        print("neutrals predicted", neutrals)
        print("entailments predicted", entailments)
        print("contradictions predicted", contradictions)
        
        return results, accuracy, neutrals, entailments, contradictions

class LogprobsEvaluator:

    @staticmethod
    def compute_logprobs(answers):
        correct_answers = []
        all_probs = []

        for answer in answers:
            """
            Note:
            - A, B and C are already dynamically mapped to their corresponding labels during inference, so:
            - Label mapping example (selected randomly): {"A": "neutral", "B": "contradiction", "C": "entailment"}
            - Corresponding unmapped "logits": {"A": -0.85, "B": -0.57, "C": 15.08},
            - And logits_mapped: {"neutral": -0.85, "contradiction": -0.57, "entailment": 15.08}
            - note that A = neutral = -0.85
            """
            logits_mapped = answer['logits_mapped']
            # label: entailment, neutral or contradiction
            label = answer['label']

            # prediction is highest logit
            pred_label = max(logits_mapped, key = logits_mapped.get)
            print('pred_label:', pred_label)
            pred_logit = logits_mapped[pred_label]

            # Determine if the prediction is correct
            is_correct = pred_label == label
            correct_answers.append(is_correct)
            all_probs.append(pred_logit)

        # Calculate the accuracy
        acc = np.mean(correct_answers)
        correct_answers = np.array(correct_answers)
        all_probs = np.array(all_probs)

        print("Average accuracy {:.3f}".format(acc))

        return correct_answers, acc, all_probs 
    

answers = [
    {
        "task": "temporal-2",
        "question_number": 1,
        "question": "Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Premise: \"Andrea was born in 1996 and Sharon was born in 2004.\". Hypothesis: \"Sharon was born after Andrea.\". Given the premise provided, is the hypothesis: A. neutral or B. contradiction or C. entailment ? \n Answer: ",
        "answer": "C. entailment. The hypothesis follows logically from the premise.",
        "logits": {
            "A": -0.8519324660301208,
            "B": -0.5721256136894226,
            "C": 15.081100463867188
        },
        "logits_mapped": {
            "neutral": -0.8519324660301208,
            "contradiction": -0.5721256136894226,
            "entailment": 15.081100463867188
        },
        "label_mapping": {
            "A": "neutral",
            "B": "contradiction",
            "C": "entailment"
        },
        "question_and_answer": "[{'role': 'user', 'content': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Premise: \"Andrea was born in 1996 and Sharon was born in 2004.\". Hypothesis: \"Sharon was born after Andrea.\". Given the premise provided, is the hypothesis: A. neutral or B. contradiction or C. entailment ? \\n Answer: '}]\n\nC. entailment. The hypothesis follows logically from the premise.",
        "instruction_and_answer": "{'A': 'neutral', 'B': 'contradiction', 'C': 'entailment'}\n\nC. entailment. The hypothesis follows logically from the premise.",
        "label": "entailment"
    },
    {
        "task": "temporal-2",
        "question_number": 2,
        "question": "Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Premise: \"Judith was born in 1991 and Linda was born in 1993.\". Hypothesis: \"Linda was born after Judith.\". Given the premise provided, is the hypothesis: A. entailment or B. contradiction or C. neutral ? \n Answer: ",
        "answer": "A. entailment. The hypothesis follows logically from the premise.",
        "logits": {
            "A": 16.58739471435547,
            "B": -1.6540007591247559,
            "C": 1.401352882385254
        },
        "logits_mapped": {
            "entailment": 16.58739471435547,
            "contradiction": -1.6540007591247559,
            "neutral": 1.401352882385254
        },
        "label_mapping": {
            "A": "entailment",
            "B": "contradiction",
            "C": "neutral"
        },
        "question_and_answer": "[{'role': 'user', 'content': 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. Premise: \"Judith was born in 1991 and Linda was born in 1993.\". Hypothesis: \"Linda was born after Judith.\". Given the premise provided, is the hypothesis: A. entailment or B. contradiction or C. neutral ? \\n Answer: '}]\n\nA. entailment. The hypothesis follows logically from the premise.",
        "instruction_and_answer": "{'A': 'entailment', 'B': 'contradiction', 'C': 'neutral'}\n\nA. entailment. The hypothesis follows logically from the premise.",
        "label": "entailment"
    },
]

evaluator = LogprobsEvaluator()
correct_answers, acc, all_probs = evaluator.compute_logprobs(answers)