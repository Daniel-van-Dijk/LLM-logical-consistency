from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from huggingface_hub import login
from models.llama3_8B import LLama3_8B



class LLama3_8B_COT(LLama3_8B):
    def inference_for_prompt(self, prompts: List[Dict[str, str]]) -> str:
        encodeds = self.tokenizer.apply_chat_template(
            prompts,
            padding=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        # remove end of generation token <|eot_id|> such that model continues after "let's think step by step"
        input_ids = encodeds['input_ids'][:, :-1]
        attention_mask = encodeds['attention_mask'][:, :-1]
        prompt_length = input_ids.size(1)
        model_inputs = {'input_ids': input_ids.to(self.model.device), 'attention_mask': attention_mask.to(self.model.device)}
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generated_dict = self.model.generate(
            **model_inputs,
            max_new_tokens=300,
            min_length=prompt_length + 10,
            eos_token_id=terminators,
            pad_token_id = self.tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        # Extract logits for the first token of each generated output
        first_token_logits_batch = [generated_dict.scores[0][i].tolist() for i in range(len(prompts))]
        return decoded_batch, first_token_logits_batch
