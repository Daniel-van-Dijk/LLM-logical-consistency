import re


def parse_yes_no_output(answers_labels):
    """ 
    Function where answer is expected to be answerable with "yes" or "no"
    example prompt: Does the hypothesis logically follow from the premise?
    We expect: "yes" if entailment, "no" if contradiction and "unsure" (or something) for neutral
    """
    correct, total = 0, 0
    contradictions, entailments, neutrals = 0, 0, 0
    results = []

    for prompt, answer, actual_label in answers_labels:

        answer_lower = answer.lower().strip()
        # look for "yes" and "no", avoid finding no as part of other words like "kNOw" 
        contains_yes = bool(re.search(r'\byes\b', answer_lower))
        contains_no = bool(re.search(r'\bno\b', answer_lower))

        if contains_yes and not contains_no:
            transformed_label = 'entailment'
            entailments += 1
        elif contains_no and not contains_yes:
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

