import re
import numpy as np
from scipy.special import softmax
import torch

def validate_args(args):
    if args.evaluation_type == 'logprobs' and args.prompt_template != 'mcq':
        raise ValueError("Log probability evaluation works only on MCQ prompt inputs.")



def parse_multiple_choice(answers_labels):
    """ 
    Function where answer is expected to be answerable with a multiple choice style
    Expected one of: (a) entailment, (b) neutral, (c) contradiction
    """
    correct, total = 0, 0
    contradictions, entailments, neutrals = 0, 0, 0
    results = []

    for prompt, answer, actual_label in answers_labels:

        answer_lower = answer.lower().strip()
        # look for "yes" and "no", avoid finding no as part of other words like "kNOw" 
        is_entailment = bool(re.search(r'\(a\)', answer_lower)) or \
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
        else:
            transformed_label = 'neutral'
            neutrals += 1

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



def compute_logprobs(answers, choices_ids):
    correct_answers = []  
    all_probs = []
    choices = ["A", "B", "C"]  

    for question, output, generated_dict, correct_answer in answers:
        lprobs = []
        for ans, choice_id in zip(choices, choices_ids):
            try:
                lprobs.append(generated_dict.logits[0][0][choice_id])
            except:
                # If an option is not in the first 1000, set its logit to -100
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)

        candidate_logits = torch.tensor(lprobs).to(torch.float32)
        print("candidate_logits:", candidate_logits)
        probs = (
            torch.nn.functional.softmax(
                candidate_logits,
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        print("probs array:", probs)
        pred = {i: k for i, k in enumerate(["A", "B", "C"])}[np.argmax(probs)]
        print("pred:", pred)
        print("correct_answer: ", correct_answer )
        correct_answer = {k: v for k, v in zip(["entailment", "neutral", "contradiction"], ["A", "B", "C"])}[correct_answer]
        print("correct_answer: ", correct_answer )

        is_correct = pred == correct_answer 
        correct_answers.append(is_correct) 
        all_probs.append(probs)

    acc = np.mean(correct_answers) 
    correct_answers = np.array(correct_answers) 
    all_probs = np.array(all_probs)

    print("Average accuracy {:.3f}".format(acc))

    return correct_answers, acc, all_probs  