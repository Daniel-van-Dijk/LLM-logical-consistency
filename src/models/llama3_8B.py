from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from huggingface_hub import login



class LLama3_8B:

    def __init__(self):
        
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )


    def inference_for_prompt(self, prompt: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            output_logits = True,
            return_dict_in_generate=True
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = self.tokenizer.decode(response, skip_special_tokens=True)
        return output, outputs
