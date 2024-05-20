from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

import torch


class Hermes13B:

    model_id: str = "NousResearch/Nous-Hermes-13b"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        # Cast to fp16 (26GB model does not fit in 20GB GPU)
        # TODO: check if half precision cause issues
        self.model = self.model.half().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


    def inference_for_prompt(self, prompt: List[Dict[str, str]]) -> str:

        encoded_prompt = self.tokenizer.apply_chat_template(
            conversation=prompt, 
            return_tensors="pt"
        ).to(self.device)
        prompt_length = encoded_prompt.size(1)

        generated_ids = self.model.generate(
            encoded_prompt,
            max_new_tokens=500,
            do_sample=True,
            output_logits = True,
            return_dict_in_generate=True
        )

        model_output_ids = generated_ids[0][prompt_length:]
        decoded_output = self.tokenizer.decode(model_output_ids, skip_special_tokens=True)

        return decoded_output, generated_ids
