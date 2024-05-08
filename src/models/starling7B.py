from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch

class Starling7B:

    model_id: str = "berkeley-nest/Starling-LM-7B-alpha"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

    def inference_for_prompt(self, prompt: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(prompt, return_tensors="pt", padding=True)  # no add_generation_prompt=True. padding=True is default or is it just needed?
        prompt_length = input_ids.size(1)
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, max_new_tokens=300, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)  # no eos_token_id?
        response = generated_ids[0][prompt_length:]
        decoded_output = self.tokenizer.decode(response)
        return decoded_output

