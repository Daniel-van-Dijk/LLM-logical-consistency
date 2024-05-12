import re


def validate_args(args):
    if args.evaluation_type == 'logprobs' and args.prompt_template != 'mcq':
        raise ValueError("Log probability evaluation works only on MCQ prompt inputs.")


# def parse_yes_no_output(answers_labels):
#     """ 
#     Function where answer is expected to be answerable with "yes" or "no"
#     example prompt: Does the hypothesis logically follow from the premise?
#     We expect: "yes" if entailment, "no" if contradiction and "unsure" (or something) for neutral
#     """
#     correct, total = 0, 0
#     contradictions, entailments, neutrals = 0, 0, 0
#     results = []

#     for prompt, answer, actual_label in answers_labels:

#         answer_lower = answer.lower().strip()
#         # look for "yes" and "no", avoid finding no as part of other words like "kNOw" 
#         contains_yes = bool(re.search(r'\byes\b', answer_lower))
#         contains_no = bool(re.search(r'\bno\b', answer_lower))

#         if contains_yes and not contains_no:
#             transformed_label = 'entailment'
#             entailments += 1
#         elif contains_no and not contains_yes:
#             transformed_label = 'contradiction'
#             contradictions += 1
#         else:
#             transformed_label = 'neutral'
#             neutrals += 1

#         total += 1
#         if transformed_label == actual_label:
#             correct += 1 

#         results.append((prompt, transformed_label, actual_label))
#     accuracy = correct / total
#     print('\n')
#     print("Accuracy:", accuracy)
#     print("neutrals predicted", neutrals)
#     print("entailments predicted", entailments)
#     print("contradictions predicted", contradictions)
    
#     return results, accuracy, neutrals, entailments, contradictions


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



def compute_logprobs(answers):
    correct = []
    all_probs = []
    choices = ["(a)", "(b)", "(c)", "(d)"]  

    for question, output, correct_answer in answers:
        lprobs = []
        for ans in choices:
            try:
                lprobs.append(output["choices"][0]["logprobs"]["top_logprobs"][-1][ans])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)
        pred = {0: "(a)", 1: "(b)", 2: "(c)", 3: "(d)"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        correct = pred == correct_answer
        correct.append(correct)
        all_probs.append(probs)

    acc = np.mean(correct)
    correct = np.array(correct)
    all_probs = np.array(all_probs)

    print("Average accuracy {:.3f}".format(acc))

    return correct, acc, all_probs