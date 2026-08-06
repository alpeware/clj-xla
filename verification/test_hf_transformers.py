import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "../.models/gemma-4-E2B"
    if not os.path.exists(model_dir):
        model_dir = "../.models/gemma-4-E2B-it"
        
    print(f"Loading tokenizer and model from local path '{model_dir}' using HuggingFace Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"\nPrompt: '{prompt}'")
    print(f"Input Token IDs: {inputs.input_ids.tolist()}")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nHuggingFace Transformers Generated Output:\n{repr(generated_text)}")

if __name__ == "__main__":
    main()
