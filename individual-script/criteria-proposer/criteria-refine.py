import json
import random
import os
from pathlib import Path
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

import time

#-----Load model----
model_path = "/users/adfy064/archive/vision_saved/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# --- Prompt Template (based on Table 10) ---
PROMPT_TEMPLATE = '''
I am a machine learning researcher working with a set of images. I aim to cluster this set of images based on the various clustering criteria present within them. Below is a preliminary list of clustering criteria that I’ve discovered to group these images:
{criterion}
My goal is to refine this list by merging similar criteria and rephrasing them using more precise and informative terms. This will help create a set of distinct, optimized clustering criteria.
Your task is to first review and understand the initial list of clustering criteria provided. Then, assist me in refining this list by: 
* Merging similar criteria. 
* Expressing each criterion more clearly and informatively.
Please respond with the cleaned and optimized list of clustering criteria, formatted as bullet points (using "*"). 
Your response:
'''

# --- Step 1: Load captions from JSONL ---
def load_criteria(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Step 2: Format a batch of captions into the prompt ---
def format_prompt(criteria: List[str]) -> str:
    criterion_block = "\n".join([f"*Criterion {i+1}: \"{input}\"" for i, input in enumerate(criteria)])
    return PROMPT_TEMPLATE.format(criterion=criterion_block)

# --- Build chat prompt using messages format ---
def build_chat_prompt(captions: List[str]) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": format_prompt(captions)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Step 3: Query LLaMA model ---
def query_llm(prompt: str, max_new_tokens: int = 4000) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.8,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("Full Response:\n", response)
    marker = "Your response:assistant"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].strip()
    else:
        return "shit!"  # fallback if marker not found

def clean_criterion(line: str) -> str:
    # Remove leading '*', trim, and remove inline Markdown bold markers
    line = line.strip()
    if line.startswith("*"):
        line = line.lstrip("* ").strip()
        # Remove all instances of '**' (Markdown bold)
        line = re.sub(r"\*\*", "", line)
    return line

# --- Step 4: Run batching and collection ---
import json
import random
from typing import List, Set

def discover_criteria(json_path: str) -> List[str]:
    # Load the full list of criteria strings
    criteria = load_criteria(json_path)

    all_criteria: Set[str] = set()

    prompt = build_chat_prompt(criteria)
    response = query_llm(prompt)

    print("Trimmed Response:\n", response)

    for line in response.split("\n"):
        if not line.strip().startswith("*"):
            continue  # Skip non-bullet lines

        cleaned = clean_criterion(line)
        if cleaned:
            all_criteria.add(cleaned)

    return sorted(all_criteria)


# --- Save criteria to JSON ---
def save_criteria(criteria_list: List[str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(criteria_list, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(criteria_list)} unique criteria to {output_path}")

# --- Step 5: Run the full pipeline ---
if __name__ == "__main__":
    input = "clustering_criteria.json"
    output = "refined_criteria.json"
    criteria_list = discover_criteria(input)

    #print("\n--- Discovered Criteria ---")
    #for c in criteria_list:
        #print(f"* {c}")
    start_time = time.time()
    save_criteria(criteria_list, output)
    end_time = time.time()

    collapsed_time = end_time - start_time
    print("[!!!] Time taken to run the whole shit: ", collapsed_time)