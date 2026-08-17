from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Hugging Face model name
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

# Confirm that 4-bit quantization is enabled
print(model.config.quantization_config)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adapters to the quantized model
model = get_peft_model(model, lora_config)

# Show trainable parameters
model.print_trainable_parameters()

# Load the training dataset
dataset = load_dataset(
    "json",
    data_files="training_data.jsonl",
    split="train"
)

# Display the dataset
print(dataset)

# Format each training example
def format_example(example):
    return {
        "text":(
            f"Question: {example['instruction']}\n"
            f"Answer: {example['output']}"
        )
    }

# Apply the formatting to the entire dataset
dataset = dataset.map(format_example)

# Display the first formatted example
print(dataset[0])