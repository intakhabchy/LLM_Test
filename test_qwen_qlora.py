# 1. Select Qwen 0.5B
# 2. Configure 4-bit quantization
# 3. Load Qwen in 4-bit
# 4. Create LoRA configuration
# 5. Add LoRA adapter
# 6. Load training dataset
# 7. Format dataset using Qwen chat template
# 8. Configure training settings
# 9. Create trainer
# 10. Train LoRA adapter
# 11. Save trained LoRA adapter

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
#    Load the model using 4-bit quantization.
#    Why? The normal model needs more memory. 4-bit representation uses much less memory.
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

# Loads Qwen using that 4-bit configuration.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

print("4-bit quantization:")
print(model.config.quantization_config)


# ==========================================
# 4. Add LoRA
# ==========================================

# This defines the LoRA adapter.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# This attaches the LoRA adapter to Qwen. ***QLoRA from here.***
model = get_peft_model(model, lora_config)

# Shows that only a small portion is trainable:
model.print_trainable_parameters()


# ==========================================
# 5. Load Training Dataset
# ==========================================

# Loads your training examples.
dataset = load_dataset(
    "json",
    data_files="training_data.jsonl",
    split="train"
)


# ==========================================
# 6. Format Dataset Using Qwen Chat Template
# ==========================================

# Converts data into Qwen's expected conversation format.
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

# Tells the trainer how training should happen.
training_args = TrainingArguments(
    output_dir="./qwen-library-lora",
    num_train_epochs=3,  # Go through the training dataset 3 times.
    per_device_train_batch_size=1,  # Process one training example at a time.
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Controls how strongly the trainable parameters are changed during learning.
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)


# ==========================================
# 8. Create Trainer
# ==========================================

# Creates the object responsible for training.
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

# This is where fine-tuning actually happens.
trainer.train()


# ==========================================
# 10. Save LoRA Adapter
# ==========================================

# This saves your trained LoRA adapter.
trainer.save_model("./qwen-library-lora")

print("\nTraining completed.")
print("LoRA adapter saved to ./qwen-library-lora")