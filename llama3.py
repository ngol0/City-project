import json
import random
from pathlib import Path
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

#-----Load model----
model_path = "/users/adfy064/archive/vision_saved/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# --- Config ---
CAPTION_FILE = "descriptions_val_set.jsonl"
SUBSET_SIZE = 400
OUTPUT_DIR = "llm_outputs"

# --- Prompt Template (based on Table 10) ---
PROMPT_TEMPLATE = '''
The following are the result of captioning a set of images:
{captions}

I am a machine learning researcher trying to figure out the potential clustering or grouping criteria that exist in these images. So I can better understand my data and group them into different clusters based on different criteria.

Come up with ten distinct clustering criteria that exist in this set of images.

Please write a list of clustering criteria (separated by bullet points “*”).

Again I want to figure out what are the potential clustering/grouping criteria that I can use to group these images into different clusters. List ten clustering or grouping criteria that often exist in this set of images based on the captioning results. Answer with a list (separated by bullet points “*”).

Your response:
'''

# --- Step 1: Load captions from JSONL ---
def load_captions(jsonl_path: str) -> List[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line)["description"] for line in f]

# --- Step 2: Format a batch of captions into the prompt ---
def format_prompt(captions: List[str]) -> str:
    caption_block = "\n".join([f"Image {i+1}: \"{caption}\"" for i, caption in enumerate(captions)])
    return PROMPT_TEMPLATE.format(captions=caption_block)

# --- Build chat prompt using messages format ---
def build_chat_prompt(captions: List[str]) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": format_prompt(captions)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Step 3: Query LLaMA model ---
def query_llm(prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "Your response:assistant"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].strip()
    else:
        return "Can't find marker!"

def clean_criterion(line: str) -> str:
    # Remove leading '*' and any ** after that
    line = line.strip()
    if line.startswith("*"):
        line = line.lstrip("* ").strip()
        # Remove all instances of '**'
        line = re.sub(r"\*\*", "", line)
    return line

# --- Step 4: Run batching ----
def discover_criteria(jsonl_path: str, batch_size: int = 400) -> List[str]:
    captions = load_captions(jsonl_path)
    random.shuffle(captions)

    all_criteria = set()

    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        prompt = build_chat_prompt(batch)
        response = query_llm(prompt)

        print(f"Batch {i // batch_size + 1} Response:\n{response}\n")

        # Extract bullet list
        for line in response.split("\n"):
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
    jsonl_file_path = "descriptions_val_set.jsonl"  # update with your actual file path
    criteria_list = discover_criteria(jsonl_file_path)
    output_path = "llama_output.json"  # path to save output

    #print("\n--- Discovered Criteria ---")
    #for c in criteria_list:
        #print(f"* {c}")

    save_criteria(criteria_list, output_path)