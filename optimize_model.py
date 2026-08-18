import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name
# )

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

# Question
question = "What is machine learning?"

# Format question
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

# Convert to tokens
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# Measure inference time
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

end_time = time.time()

# Get answer
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

# Show results
print("Answer:")
print(answer)

print("\nInference time:", round(end_time - start_time, 2), "seconds")

# Approximate model memory
model_memory = sum(
    parameter.numel() * parameter.element_size()
    for parameter in model.parameters()
)

print(
    "Model memory:",
    round(model_memory / (1024 ** 2), 2),
    "MB"
)