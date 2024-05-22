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
            #print('pred_label:', pred_label)
            pred_logit = logits_mapped[pred_label]

            # Determine if the prediction is correct
            is_correct = pred_label == label
            correct_answers.append(is_correct)
            all_probs.append(pred_logit)

        # Calculate the accuracy
        acc = np.mean(correct_answers)
        correct_answers = np.array(correct_answers)
        all_probs = np.array(all_probs)

        return correct_answers, acc, all_probs 
    