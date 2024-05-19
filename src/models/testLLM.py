from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
# class TinyTest:
#     model_id: str = "hf-internal-testing/tiny-random-PhiForCausalLM"

#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
#         self.tokenizer.pad_token = self.tokenizer.eos_token


#     def inference_for_prompt(self, prompts):
#         encodeds = self.tokenizer.apply_chat_template(prompts, return_tensors='pt', padding=True, return_dict=True)
#         prompt_length = encodeds['input_ids'].size(1)
#         print(encodeds['input_ids'].shape)
#         model_inputs = encodeds.to(self.device)
#         generated_dict = self.model.generate(**model_inputs,
#             max_new_tokens=300,
#             output_scores=False,  
#             return_dict_in_generate=False
#         )
#         generated_ids = generated_dict.sequences
#         # decode tokens starting from the index after prompt length for each prompt in the batch
#         decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        
#         # Extract logits for the first token of each generated output
#         first_token_logits_batch = [generated_dict.scores[0][i].tolist() for i in range(len(prompts))]

#         return decoded_batch, first_token_logits_batch

class ChatModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "hf-internal-testing/tiny-random-PhiForCausalLM"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def inference_for_prompt(self, prompts):
        # Encode the prompts using the chat template and pad them
        encodeds = self.tokenizer.apply_chat_template(prompts, return_tensors='pt', padding=True, return_dict=True)
        prompt_length = encodeds['input_ids'].size(1)
        print(encodeds['input_ids'].shape)
        model_inputs = encodeds.to(self.device)
        generated_dict = self.model.generate(
            **model_inputs,
            max_new_tokens=300,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        print(len(generated_ids[0]))
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        # Extract logits for the first token of each generated output
        first_token_logits_batch = [generated_dict.scores[0][i].tolist() for i in range(len(prompts))]

        return decoded_batch, first_token_logits_batch

if __name__ == "__main__":
    chat_model = ChatModel()
    messages = [
        [
            {"role": "user", "content": "What is your favourite condiment?"},
            {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
            {"role": "user", "content": "Do you have mayonnaise recipes?"}
        ],
        [
            {"role": "user", "content": "What's your favorite color?"},
            {"role": "assistant", "content": "I love the color blue. It reminds me of the ocean and the sky, and it's very calming."},
            {"role": "user", "content": "Can you tell me a joke?"}
        ]
    ]
    responses, logits = chat_model.inference_for_prompt(messages)
    for response in responses:
        print(len(response))
    for logit in logits:
        print(len(logit))