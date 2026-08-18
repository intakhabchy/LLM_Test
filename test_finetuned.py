from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./qwen-library-lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name
)

# Load trained LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

# Question
question = "How many books can a member borrow?"

# Use Qwen's chat format
messages = [
    {
        "role": "user",
        "content": question
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

# Decode only the newly generated tokens
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("Question:", question)
print("Answer:", answer)