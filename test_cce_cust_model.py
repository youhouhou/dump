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
    
    # Get peak memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return result, peak_memory_mb

def run_benchmark(model_name, token_lengths):
    """Run benchmark for different token lengths, loading one model at a time."""
    
    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Data collection lists
    original_memory = []
    cce_memory = []
    actual_lengths = []
    
    # Base text for testing
    base_text = "This is a test of the cross entropy memory usage for benchmarking purposes. "
    
    # Test each requested token length
    for length in token_lengths:
        try:
            print(f"\n===== Testing with target length: {length} tokens =====")
            
            # Create a text that's approximately the right length
            repetitions = max(1, length // len(tokenizer.encode(base_text)))
            test_text = base_text * repetitions
            
            # Encode and truncate to desired length
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=length)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get actual token length
            actual_length = inputs["input_ids"].size(1)
            print(f"Actual sequence length: {actual_length} tokens")
            actual_lengths.append(actual_length)
            
            # Create labels
            labels = inputs["input_ids"].clone()
            
            # ===== Test original model =====
            print("Loading original LlamaForCausalLM model...")
            original_model = transformers.LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            def standard_forward():
                with torch.no_grad():
                    outputs = original_model(**inputs, labels=labels)
                return outputs
            
            _, std_mem = measure_memory(standard_forward)
            original_memory.append(std_mem)
            print(f"Original model memory: {std_mem:.2f} MB")
            
            # Clean up original model
            del original_model
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)  # Give some time for memory to be freed
            
            # ===== Test CCE model =====
            print("Loading custom LlamaForCausalLM_CCE model...")
            cce_model = LlamaForCausalLM_CCE.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            def cce_forward():
                with torch.no_grad():
                    outputs = cce_model(**inputs, labels=labels)
                return outputs
            
            _, cce_mem = measure_memory(cce_forward)
            cce_memory.append(cce_mem)
            print(f"CCE model memory: {cce_mem:.2f} MB")
            
            # Clean up CCE model
            del cce_model
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)  # Give some time for memory to be freed
            
            # Calculate and show savings
            savings = std_mem - cce_mem
            percent_saved = (savings / std_mem) * 100
            print(f"Memory saved: {savings:.2f} MB ({percent_saved:.2f}%)")
            
        except RuntimeError as e:
            print(f"Error at length {length}: {e}")
            # Keep the results we've collected so far
            break
    
    return actual_lengths, original_memory, cce_memory

def plot_results(lengths, original_memory, cce_memory):
    plt.figure(figsize=(12, 8))
    
    # Filter out any None values
    valid_indices = [i for i, (o, c) in enumerate(zip(original_memory, cce_memory)) 
                     if o is not None and c is not None]
    valid_lengths = [lengths[i] for i in valid_indices]
    valid_orig = [original_memory[i] for i in valid_indices]
    valid_cce = [cce_memory[i] for i in valid_indices]
    
    plt.plot(valid_lengths, valid_orig, 'o-', label='Original LlamaForCausalLM', linewidth=2)
    plt.plot(valid_lengths, valid_cce, 's-', label='LlamaForCausalLM_CCE', linewidth=2)
    
    # Calculate the memory savings
    savings = [orig - cce for orig, cce in zip(valid_orig, valid_cce)]
    percent_savings = [(orig - cce) / orig * 100 for orig, cce in zip(valid_orig, valid_cce)]
    
    # Add a third line for the savings percentage
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(valid_lengths, percent_savings, 'g--', label='Memory Savings %', linewidth=2)
    ax2.set_ylabel('Memory Savings (%)', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Set up the plot
    plt.title('Memory Usage vs Sequence Length', fontsize=16)
    ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax1.set_ylabel('GPU Memory Usage (MB)', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Annotate with actual values at a few points
    for i in range(0, len(valid_lengths), max(1, len(valid_lengths) // 5)):
        ax1.annotate(f"{valid_orig[i]:.1f} MB", 
                     (valid_lengths[i], valid_orig[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
        
        ax1.annotate(f"{valid_cce[i]:.1f} MB", 
                     (valid_lengths[i], valid_cce[i]), 
                     textcoords="offset points", 
                     xytext=(0,-15), 
                     ha='center')
        
        ax2.annotate(f"{percent_savings[i]:.1f}%", 
                     (valid_lengths[i], percent_savings[i]), 
                     textcoords="offset points", 
                     xytext=(10,0), 
                     ha='left',
                     color='g')
    
    plt.tight_layout()
    plt.savefig('memory_usage_comparison.png', dpi=300)
    print("Plot saved as 'memory_usage_comparison.png'")
    plt.show()

def main():
    # Model name - using Llama or another model you have access to
    model_name = "meta-llama/Llama-2-7b-hf"  # Change as needed
    
    # Define sequence lengths to test
    # Start small and increase gradually
    token_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # Optional: add more extreme lengths if your GPU can handle it
    if torch.cuda.get_device_properties(0).total_memory > 24 * 1024**3:  # More than 24GB VRAM
        token_lengths.extend([8192, 16384])
    
    print(f"Running benchmark with token lengths: {token_lengths}")
    lengths, original_memory, cce_memory = run_benchmark(model_name, token_lengths)
    
    print("\nResults summary:")
    for l, o, c in zip(lengths, original_memory, cce_memory):
        if o is not None and c is not None:
            savings = o - c
            percent = (savings / o) * 100
            print(f"Length {l}: Original {o:.2f} MB, CCE {c:.2f} MB, Savings {savings:.2f} MB ({percent:.2f}%)")
    
    # Plot the results
    plot_results(lengths, original_memory, cce_memory)

if __name__ == "__main__":
    main()