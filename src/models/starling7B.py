from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch

class Starling7B:

    model_id: str = "berkeley-nest/Starling-LM-7B-alpha"
        
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


    def inference_for_prompt(self, prompt: List[Dict[str, str]]) -> str:
        single_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        input_ids = self.tokenizer(single_turn_prompt, return_tensors="pt").input_ids
        prompt_length = input_ids.size(1)
        input_ids = input_ids.to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_length=256,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids[prompt_length:], skip_special_tokens=True)
        return response_text

