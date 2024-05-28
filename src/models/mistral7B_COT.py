from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from models.mistral7B import Mistral7B
from huggingface_hub import login

# make token on hugging face and insert here
# huggingface_token = "insert_token"
# login(token=huggingface_token)

class Mistral7B_COT(Mistral7B):
    def inference_for_prompt(self, prompts):
        encodeds = self.tokenizer.apply_chat_template(prompts, return_tensors='pt', padding=True, return_dict=True)
        # Remove the end of generation token </s> such that model continues generating after "let's think step by step"
        input_ids = encodeds['input_ids'][:, :-1]
        attention_mask = encodeds['attention_mask'][:, :-1]
        prompt_length = input_ids.size(1)

        model_inputs = {'input_ids': input_ids.to(self.device), 'attention_mask': attention_mask.to(self.device)}
        generated_dict = self.model.generate(
            **model_inputs,
            max_new_tokens=300,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        # Extract logits for the first token of each generated output
        first_token_logits_batch = [generated_dict.scores[0][i].tolist() for i in range(len(prompts))]
        return decoded_batch, first_token_logits_batch
