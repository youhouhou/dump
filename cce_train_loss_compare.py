import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from cut_cross_entropy.transformers import cce_patch

# Set seed for reproducibility
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Helper function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"GPU memory cleared")

# Helper function to print memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Configuration
MODEL_NAME = "meta-llama/Llama-3-8B"  # Use smaller model if needed, like TinyLlama
OUTPUT_DIR = "./llama_sft_lora_experiment"
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4  # LoRA typically uses slightly higher learning rates
MAX_SEQ_LENGTH = 512
NUM_SAMPLES = 200  # Small subset for quick comparison

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset from local disk
print("Loading dataset from disk...")
train_dataset = load_from_disk("./local_summarize_dataset")
# If you need to filter to just a subset:
train_dataset = train_dataset["train"].select(range(min(NUM_SAMPLES, len(train_dataset["train"]))))

# Preprocess data
def preprocess_function(examples):
    # Format as simple instruction following
    texts = []
    for post, summary in zip(examples["post"], examples["summary"]):
        texts.append(f"Summarize the following text:\n\n{post}\n\nSummary: {summary}")
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create labels (for causal language modeling)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Tokenize dataset
print("Tokenizing dataset...")
tokenized_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define LoRA configuration
def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

# Define custom logging callback
class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.training_loss = []
        self.training_steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_loss.append(logs["loss"])
            self.training_steps.append(state.global_step)

# Define training arguments
def get_training_args(model_type):
    return TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, model_type),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
        fp16=True,  # Enable mixed precision
        seed=SEED,  # Ensure same seed used for both runs
    )

# =================== TRAIN ORIGINAL MODEL (with LoRA) ===================
print("=" * 50)
print("PHASE 1: TRAINING ORIGINAL MODEL WITH LORA")
print("=" * 50)

# Load the original model and prepare for LoRA
print("Loading original model...")
original_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,
    device_map="auto"
)
print_gpu_memory()

# Apply LoRA to original model
print("Applying LoRA to original model...")
original_lora_config = get_lora_config()
original_model = get_peft_model(original_model, original_lora_config)
original_model.print_trainable_parameters()

# Train the original model
print("Training original model...")
original_callback = LossLoggingCallback()
original_training_args = get_training_args("original_lora")

original_trainer = Trainer(
    model=original_model,
    args=original_training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[original_callback]
)

original_trainer.train()
original_losses = original_callback.training_loss

# Save original model losses
torch.save(
    {'original_losses': original_losses},
    os.path.join(OUTPUT_DIR, 'original_lora_losses.pt')
)

# Clear memory
del original_model
del original_trainer
clear_gpu_memory()
print_gpu_memory()

# =================== TRAIN PATCHED MODEL (with LoRA) ===================
print("\n" + "=" * 50)
print("PHASE 2: TRAINING PATCHED MODEL WITH LORA")
print("=" * 50)

# Load model again for patched version
print("Loading model for patching...")
patched_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print_gpu_memory()

# Apply the CCE patch
print("Applying CCE patch...")
patched_model = cce_patch(patched_model, impl="cce_exact")  # Using cce_exact for accuracy
print("Patch applied successfully!")

# Apply LoRA to patched model
print("Applying LoRA to patched model...")
patched_lora_config = get_lora_config()
patched_model = get_peft_model(patched_model, patched_lora_config)
patched_model.print_trainable_parameters()

# Train the patched model
print("Training patched model...")
patched_callback = LossLoggingCallback()
patched_training_args = get_training_args("patched_lora")

patched_trainer = Trainer(
    model=patched_model,
    args=patched_training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[patched_callback]
)

patched_trainer.train()
patched_losses = patched_callback.training_loss

# Save patched model losses
torch.save(
    {'patched_losses': patched_losses},
    os.path.join(OUTPUT_DIR, 'patched_lora_losses.pt')
)

# Clear memory
del patched_model
del patched_trainer
clear_gpu_memory()
print_gpu_memory()

# =================== ANALYSIS ===================
print("\n" + "=" * 50)
print("PHASE 3: ANALYSIS")
print("=" * 50)

# Load both sets of losses
original_data = torch.load(os.path.join(OUTPUT_DIR, 'original_lora_losses.pt'))
patched_data = torch.load(os.path.join(OUTPUT_DIR, 'patched_lora_losses.pt'))

original_losses = original_data['original_losses']
patched_losses = patched_data['patched_losses']

# Save combined losses
torch.save({
    'original_losses': original_losses,
    'patched_losses': patched_losses
}, os.path.join(OUTPUT_DIR, 'combined_lora_loss_data.pt'))

# Find the minimum length to compare (in case one run had fewer steps)
min_length = min(len(original_losses), len(patched_losses))
original_losses = original_losses[:min_length]
patched_losses = patched_losses[:min_length]

# Plot loss comparison (enhanced version)
plt.figure(figsize=(12, 6))
plt.plot(original_losses, label='Original Cross-Entropy', color='blue', alpha=0.7)
plt.plot(patched_losses, label='Patched Cross-Entropy (cce_exact)', color='red', alpha=0.7)

# Add smoothed lines if there are enough data points
if min_length > 50:
    window_size = min(50, min_length // 10)  # Adjust window size based on data amount
    
    # Simple moving average for smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    original_smooth = smooth(original_losses, window_size)
    patched_smooth = smooth(patched_losses, window_size)
    
    # Calculate x-positions for smoothed data
    x_smooth = range(window_size-1, min_length)
    
    # Plot smoothed lines
    plt.plot(x_smooth, original_smooth, label='Original (Smoothed)', color='blue', linestyle='--')
    plt.plot(x_smooth, patched_smooth, label='Patched (Smoothed)', color='red', linestyle='--')

plt.title('Training Loss Comparison (LoRA)', fontsize=14)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lora_loss_comparison.png'), dpi=300)
plt.close()

# Calculate statistics
loss_diff = [o - p for o, p in zip(original_losses, patched_losses)]
avg_diff = sum(loss_diff) / len(loss_diff)
max_diff = max(loss_diff)
min_diff = min(loss_diff)

print(f"Loss difference statistics:")
print(f"Average difference: {avg_diff:.6f}")
print(f"Maximum difference: {max_diff:.6f}")
print(f"Minimum difference: {min_diff:.6f}")

# Plot loss difference
plt.figure(figsize=(10, 6))
plt.plot(loss_diff)
plt.title('Loss Difference (Original - Patched) with LoRA')
plt.xlabel('Steps')
plt.ylabel('Difference')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'lora_loss_difference.png'), dpi=300)
plt.close()

# Add an additional visualization to better highlight convergence differences
plt.figure(figsize=(12, 6))

# Plot the ratio of losses (can highlight relative performance differences)
loss_ratio = [p/o if o > 0 else 1.0 for o, p in zip(original_losses, patched_losses)]
plt.plot(loss_ratio, label='Loss Ratio (Patched/Original)', color='purple')
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
plt.title('Relative Performance: Patched vs Original Model with LoRA', fontsize=14)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Loss Ratio', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lora_loss_ratio.png'), dpi=300)
plt.close()

# Save the analysis results in a text file
with open(os.path.join(OUTPUT_DIR, 'lora_analysis_results.txt'), 'w') as f:
    f.write("Loss Comparison Analysis with LoRA\n")
    f.write("==================================\n\n")
    f.write(f"Number of steps compared: {min_length}\n")
    f.write(f"Average original loss: {sum(original_losses)/min_length:.6f}\n")
    f.write(f"Average patched loss: {sum(patched_losses)/min_length:.6f}\n")
    f.write(f"Average loss difference: {avg_diff:.6f}\n")
    f.write(f"Maximum loss difference: {max_diff:.6f}\n")
    f.write(f"Minimum loss difference: {min_diff:.6f}\n")
    
    # Calculate and add final loss difference
    final_diff_pct = ((original_losses[-1] - patched_losses[-1]) / original_losses[-1]) * 100 if original_losses[-1] != 0 else 0
    f.write(f"Final loss values - Original: {original_losses[-1]:.6f}, Patched: {patched_losses[-1]:.6f}\n")
    f.write(f"Final loss difference: {original_losses[-1] - patched_losses[-1]:.6f} ({final_diff_pct:.2f}%)\n")

print("Enhanced LoRA analysis completed successfully!")