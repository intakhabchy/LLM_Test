from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import torch


# ==========================================
# 1. Model
# ==========================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"


# ==========================================
# 2. 4-bit Quantization
# ==========================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


# ==========================================
# 3. Load Tokenizer and Model
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

print("4-bit quantization:")
print(model.config.quantization_config)


# ==========================================
# 4. Add LoRA
# ==========================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# ==========================================
# 5. Load Training Dataset
# ==========================================

dataset = load_dataset(
    "json",
    data_files="training_data.jsonl",
    split="train"
)


# ==========================================
# 6. Format Dataset Using Qwen Chat Template
# ==========================================

def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }


dataset = dataset.map(format_chat)

print("\nFirst training example:")
print(dataset[0]["text"])


# ==========================================
# 7. Training Configuration
# ==========================================

training_args = TrainingArguments(
    output_dir="./qwen-library-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)


# ==========================================
# 8. Create Trainer
# ==========================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)


# ==========================================
# 9. Train
# ==========================================

print("\nStarting training...")

trainer.train()


# ==========================================
# 10. Save LoRA Adapter
# ==========================================

trainer.save_model("./qwen-library-lora")

print("\nTraining completed.")
print("LoRA adapter saved to ./qwen-library-lora")