from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from huggingface_hub import login

# make token on hugging face and insert here
# huggingface_token = "insert_token"
# login(token=huggingface_token)

class TinyTest:
    model_id: str = "hf-internal-testing/tiny-random-PhiForCausalLM"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def inference_for_prompt(self, prompts: List[List[Dict[str, str]]]) -> List[str]:
        encodeds = self.tokenizer.apply_chat_template(prompts, return_tensors="pt", padding=True)
        prompt_length = encodeds.size(1)
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=300, pad_token_id=self.tokenizer.eos_token_id)
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        return decoded_batch
