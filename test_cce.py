import torch
import transformers
from cut_cross_entropy.transformers import cce_patch
import gc

def measure_memory(func):
    """Run a function and measure its peak memory usage."""
    # Clear cache to get accurate measurements
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Run the function
    result = func()
    
    # Get peak memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return result, peak_memory_mb

def main():
    # Load a model - use a model with large vocabulary for more dramatic results
    # For example, Gemma2 or Llama would show larger differences
    model_name = "google/gemma-2b"  # or "meta-llama/Llama-2-7b-hf" if you have access
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,  # Use mixed precision for efficiency
        device_map="auto"
    )
    
    # Create some sample inputs
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Sample text with labels
    inputs = tokenizer("This is a test of the cross entropy memory usage. Let's see how much memory we save.", 
                      return_tensors="pt").to(model.device)
    
    # Create labels for loss computation (shifted right by 1)
    labels = inputs["input_ids"].clone()
    
    # 1. Measure standard forward pass with loss computation
    def standard_forward():
        with torch.no_grad():  # No need for gradients in this test
            outputs = model(**inputs, labels=labels)
        return outputs
    
    _, standard_memory = measure_memory(standard_forward)
    print(f"Standard forward pass peak memory: {standard_memory:.2f} MB")
    
    # 2. Apply CCE patch and measure again
    model = cce_patch(model, reduction="mean")
    
    def cce_forward():
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        return outputs
    
    _, cce_memory = measure_memory(cce_forward)
    print(f"CCE patched forward pass peak memory: {cce_memory:.2f} MB")
    
    # Calculate memory savings
    savings = standard_memory - cce_memory
    percent_saved = (savings / standard_memory) * 100
    
    print(f"\nMemory saved: {savings:.2f} MB ({percent_saved:.2f}%)")

if __name__ == "__main__":
    main()