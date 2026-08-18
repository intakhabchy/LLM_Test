# 1. Select Qwen 0.5B
# 2. Load tokenizer
# 3. Load original Qwen model
# 4. Load trained LoRA adapter
# 5. Attach adapter to Qwen
# 6. Give the model a question
# 7. Format question using Qwen chat template
# 8. Convert question to tokens
# 9. Generate answer
# 10. Decode tokens into text
# 11. Display answer

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Use the original Qwen model and this trained adapter.
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

# Converts it into Qwen's chat format.
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize - Converts the prompt into tokens.
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# Generate - Generate an answer
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

# Decode only the newly generated tokens
# Removes the original question from the output.
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

# Converts the generated tokens back into normal text.
answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

# Shows you the result.
print("Question:", question)
print("Answer:", answer)