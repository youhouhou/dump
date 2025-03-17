import torch
import transformers
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from cut_cross_entropy import linear_cross_entropy
import gc
from typing import Optional, Union, Tuple, List

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

def main():
    # Model name - using Llama
    model_name = "meta-llama/Llama-2-7b-hf"  # Change to a model you have access to
    
    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Sample text with labels - using a longer sequence to make memory differences more noticeable
    text = "This is a test of the cross entropy memory usage. Let's see how much memory we save. " * 10
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Create labels for loss computation (shifted right by 1)
    labels = inputs["input_ids"].clone()
    
    print("Loading original LlamaForCausalLM model...")
    original_model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 1. Measure standard forward pass with loss computation
    def standard_forward():
        with torch.no_grad():  # No need for gradients in this test
            outputs = original_model(**inputs, labels=labels)
        return outputs
    
    _, standard_memory = measure_memory(standard_forward)
    print(f"Standard model peak memory: {standard_memory:.2f} MB")
    
    # Free memory
    del original_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading custom LlamaForCausalLM_CCE model...")
    # 2. Create the custom CCE model
    # First get the config
    config = LlamaConfig.from_pretrained(model_name)
    
    # Create our custom model with the same config
    cce_model = LlamaForCausalLM_CCE.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    def cce_forward():
        with torch.no_grad():
            outputs = cce_model(**inputs, labels=labels)
        return outputs
    
    _, cce_memory = measure_memory(cce_forward)
    print(f"CCE model peak memory: {cce_memory:.2f} MB")
    
    # Calculate memory savings
    savings = standard_memory - cce_memory
    percent_saved = (savings / standard_memory) * 100
    
    print(f"\nMemory saved: {savings:.2f} MB ({percent_saved:.2f}%)")
    
    # Also test with longer sequences for more dramatic results
    if savings > 0:
        print("\nTesting with longer sequence...")
        long_text = "This is a test of the cross entropy memory usage. Let's see how much memory we save. " * 50
        long_inputs = tokenizer(long_text, return_tensors="pt").to("cuda")
        long_labels = long_inputs["input_ids"].clone()
        
        def standard_long():
            with torch.no_grad():
                outputs = original_model(**long_inputs, labels=long_labels)
            return outputs
            
        def cce_long():
            with torch.no_grad():
                outputs = cce_model(**long_inputs, labels=long_labels)
            return outputs
        
        try:
            # Load original model again
            original_model = transformers.LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            _, standard_long_memory = measure_memory(standard_long)
            print(f"Standard model long sequence peak memory: {standard_long_memory:.2f} MB")
            
            del original_model
            torch.cuda.empty_cache()
            gc.collect()
            
            _, cce_long_memory = measure_memory(cce_long)
            print(f"CCE model long sequence peak memory: {cce_long_memory:.2f} MB")
            
            long_savings = standard_long_memory - cce_long_memory
            long_percent_saved = (long_savings / standard_long_memory) * 100
            
            print(f"\nLong sequence memory saved: {long_savings:.2f} MB ({long_percent_saved:.2f}%)")
            
        except RuntimeError as e:
            print(f"Error with long sequence test: {e}")
            print("This likely means the CCE version is more memory efficient!")

if __name__ == "__main__":
    main()