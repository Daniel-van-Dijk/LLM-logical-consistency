from typing import List, Dict
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Mistral7B_ft:
    model_id: str = "mistralai/Mistral-7B-v0.1"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def inference_for_prompt(self, prompt: List[Dict[str, str]]) -> str:
        encodeds = self.tokenizer.apply_chat_template(prompt, return_tensors="pt", padding=True)
        prompt_length = encodeds.size(1)
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=300, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.decode(generated_ids[0][prompt_length:])
        return decoded
