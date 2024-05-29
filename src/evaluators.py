from typing import List
import numpy as np

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
    