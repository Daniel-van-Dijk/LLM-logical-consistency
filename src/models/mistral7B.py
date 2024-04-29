from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# make token on hugging face and insert here
huggingface_token = "insert_token"
login(token=huggingface_token)

device = "cuda" 

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "Matthew is excellent and experienced.	Matthew is excellent. a: entailment, b: contradiction or c: neutral?"},
    {"role": "assistant", "content": "a"},
    {"role": "user", "content": "Dave is fitter than Andrea. Andrea is fitter than Dave. a: entailment, b: contradiction or c: neutral?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])