import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./qwen-library-lora"

# Load evaluation dataset
with open("evaluation_data.jsonl","r") as file:
    evaluation_data = [
        json.loads(line)
        for line in file
    ]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load original model
original_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load fine-tuned model
fine_tuned_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(model_name),
    adapter_path
)

def generate_answer(model, question):

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

# Compare the two models
for item in evaluation_data:

    question = item["question"]
    expected = item["answer"]

    original_answer = generate_answer(
        original_model,
        question
    )

    fine_tuned_answer = generate_answer(
        fine_tuned_model,
        question
    )

    print("\nQuestion:", question)
    print("Expected:", expected)
    print("Original:", original_answer)
    print("Fine-tuned:", fine_tuned_answer)
    print("-" * 60)