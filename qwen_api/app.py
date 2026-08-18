from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded!")

@app.get("/ask")
def ask(question: str):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )

    inputs = tokenizer(
        prompt,
        return_tensors = "pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 30
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens = True
    )

    return {
        "question": question,
        "answer": answer
    }