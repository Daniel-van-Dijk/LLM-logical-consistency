from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch

class Starling7B:

    model_id: str = "berkeley-nest/Starling-LM-7B-alpha"
        
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = 'left'


    def inference_for_prompt(self, prompts):
        encodeds = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompt_length = encodeds['input_ids'].size(1)
        model_inputs = encodeds.to(self.device)
        generated_dict = self.model.generate(
            **model_inputs,
            max_length=256,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        # Extract logits for the first token of each generated output
        first_token_logits_batch = [generated_dict.scores[0][i].tolist() for i in range(len(prompts))]
        return decoded_batch, first_token_logits_batch