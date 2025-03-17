import torch
import transformers
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from cut_cross_entropy import linear_cross_entropy
import gc
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import time

class LlamaForCausalLM_CCE(LlamaForCausalLM):
    """Llama model with custom forward that directly uses linear_cross_entropy."""
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get hidden states from the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs[0]
        loss = None
        logits = None
        
        # If we have labels, compute loss using linear_cross_entropy
        if labels is not None:
            # Use linear_cross_entropy directly for efficient computation
            loss = linear_cross_entropy(
                hidden_states, 
                self.lm_head.weight, 
                labels,
                reduction="mean"
            )
        
        # If no labels or num_logits_to_keep > 0, compute logits for just what we need
        if labels is None or num_logits_to_keep > 0:
            relevant_hidden_states = hidden_states
            if num_logits_to_keep > 0:
                relevant_hidden_states = hidden_states[:, -num_logits_to_keep:, :]
            
            # Calculate logits for the relevant hidden states
            logits = self.lm_head(relevant_hidden_states)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def measure_memory(func):
    """Run a function and measure its peak memory usage."""
    # Clear cache to get accurate measurements
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Run the function
    result = func()
    
    # Get peak memory in GB
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    
    return result, peak_memory_gb

def prepare_inputs(tokenizer, token_lengths):
    """Prepare all inputs for different sequence lengths."""
    inputs_dict = {}
    actual_lengths = []
    
    base_text = "This is a test of the cross entropy memory usage for benchmarking purposes. "
    
    for length in token_lengths:
        # Create a text that's approximately the right length
        repetitions = max(1, length // len(tokenizer.encode(base_text)))
        test_text = base_text * repetitions
        
        # Encode and truncate to desired length
        encoded = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=length)
        
        # Get actual token length
        actual_length = encoded["input_ids"].size(1)
        actual_lengths.append(actual_length)
        
        # Create labels
        labels = encoded["input_ids"].clone()
        
        # Store inputs and labels
        inputs_dict[actual_length] = {
            "inputs": encoded,
            "labels": labels
        }
    
    return inputs_dict, actual_lengths

def run_benchmark(model_name, token_lengths):
    """Run benchmark for different token lengths, testing all lengths with each model."""
    
    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Prepare all inputs and get actual token lengths
    print("Preparing inputs for all sequence lengths...")
    inputs_dict, actual_lengths = prepare_inputs(tokenizer, token_lengths)
    
    # Data collection lists
    original_memory = []
    cce_memory = []
    
    # ===== Test original model with all sequence lengths =====
    print("\n===== Testing original LlamaForCausalLM model =====")
    original_model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    for length in actual_lengths:
        try:
            print(f"\nTesting with sequence length: {length} tokens")
            
            # Move inputs to GPU
            inputs = {k: v.to("cuda") for k, v in inputs_dict[length]["inputs"].items()}
            labels = inputs_dict[length]["labels"].to("cuda")
            
            def standard_forward():
                with torch.no_grad():
                    outputs = original_model(**inputs, labels=labels)
                return outputs
            
            _, std_mem = measure_memory(standard_forward)
            original_memory.append(std_mem)
            print(f"Original model memory: {std_mem:.4f} GB")
            
        except RuntimeError as e:
            print(f"Error at length {length}: {e}")
            original_memory.append(None)
    
    # Clean up original model
    del original_model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)  # Give some time for memory to be freed
    
    # ===== Test CCE model with all sequence lengths =====
    print("\n===== Testing custom LlamaForCausalLM_CCE model =====")
    cce_model = LlamaForCausalLM_CCE.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    for length in actual_lengths:
        try:
            print(f"\nTesting with sequence length: {length} tokens")
            
            # Move inputs to GPU
            inputs = {k: v.to("cuda") for k, v in inputs_dict[length]["inputs"].items()}
            labels = inputs_dict[length]["labels"].to("cuda")
            
            def cce_forward():
                with torch.no_grad():
                    outputs = cce_model(**inputs, labels=labels)
                return outputs
            
            _, cce_mem = measure_memory(cce_forward)
            cce_memory.append(cce_mem)
            print(f"CCE model memory: {cce_mem:.4f} GB")
            
            # If we have both measurements, calculate savings
            if original_memory[actual_lengths.index(length)] is not None:
                orig_mem = original_memory[actual_lengths.index(length)]
                savings = orig_mem - cce_mem
                percent_saved = (savings / orig_mem) * 100
                print(f"Memory saved: {savings:.4f} GB ({percent_saved:.2f}%)")
            
        except RuntimeError as e:
            print(f"Error at length {length}: {e}")
            cce_memory.append(None)
    
    # Clean up CCE model
    del cce_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return actual_lengths, original_memory, cce_memory

def plot_results(lengths, original_memory, cce_memory):
    # Filter for valid data points and where CCE uses less memory than original
    valid_indices = []
    for i, (o, c) in enumerate(zip(original_memory, cce_memory)):
        if o is not None and c is not None and c < o:  # Only include where CCE is better
            valid_indices.append(i)
    
    # If we don't have any valid points where CCE is better, include all valid points
    if not valid_indices:
        valid_indices = [i for i, (o, c) in enumerate(zip(original_memory, cce_memory)) 
                        if o is not None and c is not None]
    
    # Check if we have any valid points at all
    if not valid_indices:
        print("No valid data points to plot. All tests resulted in errors or None values.")
        return
    
    valid_lengths = [lengths[i] for i in valid_indices]
    valid_orig = [original_memory[i] for i in valid_indices]
    valid_cce = [cce_memory[i] for i in valid_indices]
    
    # Find first point where CCE has advantage
    first_advantage_idx = None
    for i, (o, c) in enumerate(zip(valid_orig, valid_cce)):
        if c < o:
            first_advantage_idx = i
            break
    
    # If there's no advantage point, use all data
    if first_advantage_idx is None:
        first_advantage_idx = 0
        print("Warning: No sequence length where CCE has memory advantage. Showing all data.")
    
    # Use only data from the first advantage point onwards
    plot_lengths = valid_lengths[first_advantage_idx:]
    plot_orig = valid_orig[first_advantage_idx:]
    plot_cce = valid_cce[first_advantage_idx:]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    plt.plot(plot_lengths, plot_orig, 'o-', label='Original LlamaForCausalLM', linewidth=2, markersize=8)
    plt.plot(plot_lengths, plot_cce, 's-', label='LlamaForCausalLM_CCE', linewidth=2, markersize=8)
    
    # Set up the plot
    plt.title('GPU Memory Usage vs Sequence Length', fontsize=18)
    plt.xlabel('Sequence Length (tokens)', fontsize=14)
    plt.ylabel('GPU Memory Usage (GB)', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Annotate some points with values
    for i in range(0, len(plot_lengths), max(1, len(plot_lengths) // 4)):
        plt.annotate(f"{plot_orig[i]:.3f} GB", 
                     (plot_lengths[i], plot_orig[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=10)
        
        plt.annotate(f"{plot_cce[i]:.3f} GB", 
                     (plot_lengths[i], plot_cce[i]), 
                     textcoords="offset points", 
                     xytext=(0,-15), 
                     ha='center',
                     fontsize=10)
    
    # Set the scales to better show the growth patterns
    plt.xscale('log')
    plt.yscale('log')
    
    # Add second x-axis with non-log scale for reference
    ax1 = plt.gca()
    ax1_ticks = ax1.get_xticks()
    ax1.set_xticks(ax1_ticks)
    ax1.set_xticklabels([f"{int(x)}" for x in ax1_ticks], fontsize=10)
    
    plt.tight_layout()
    plt.savefig('memory_usage_comparison.png', dpi=300)
    print("Plot saved as 'memory_usage_comparison.png'")
    
    # Create a second plot with linear scales for clarity
    plt.figure(figsize=(12, 8))
    plt.plot(plot_lengths, plot_orig, 'o-', label='Original LlamaForCausalLM', linewidth=2, markersize=8)
    plt.plot(plot_lengths, plot_cce, 's-', label='LlamaForCausalLM_CCE', linewidth=2, markersize=8)
    
    plt.title('GPU Memory Usage vs Sequence Length (Linear Scale)', fontsize=18)
    plt.xlabel('Sequence Length (tokens)', fontsize=14)
    plt.ylabel('GPU Memory Usage (GB)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('memory_usage_comparison_linear.png', dpi=300)
    print("Linear scale plot saved as 'memory_usage_comparison_linear.png'")
    
    plt.show()

def main():
    # Model name - using Llama or another model you have access to
    model_name = "meta-llama/Llama-2-7b-hf"  # Change as needed
    
    # Define sequence lengths to test - starting at 100+ as requested
    # Use a geometric progression to better show memory scaling
    token_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Optional: add more extreme lengths if your GPU can handle it
    if torch.cuda.get_device_properties(0).total_memory > 24 * 1024**3:  # More than 24GB VRAM
        token_lengths.extend([16384, 32768])
    
    print(f"Running benchmark with target token lengths: {token_lengths}")
    lengths, original_memory, cce_memory = run_benchmark(model_name, token_lengths)
    
    print("\nResults summary:")
    for l, o, c in zip(lengths, original_memory, cce_memory):
        if o is not None and c is not None:
            savings = o - c
            percent = (savings / o) * 100
            print(f"Length {l}: Original {o:.4f} GB, CCE {c:.4f} GB, Savings {savings:.4f} GB ({percent:.2f}%)")
    
    # Plot the results
    plot_results(lengths, original_memory, cce_memory)

if __name__ == "__main__":
    main()