import json
from typing import List, Tuple, Dict
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

import time
from collections import defaultdict
import json

#-----Load model----
model_path = "/users/adfy064/archive/vision_saved/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set left padding for decoder-only models (CRITICAL for batch generation)
tokenizer.padding_side = 'left'

# --- Config ---
CAPTION_FILE = "criterion_captions.jsonl"
OUTPUT_DIR = ".test/llm_outputs_grouper_a100.json"

# --- Prompt Template (based on Table 17) ---
PROMPT_TEMPLATE = '''
The following is the description about the "{criterion}" of an image:

{criterion_specific_caption}

I am a machine learning researcher trying to assign a label to this image based on what is the "{criterion}" depicted in this image.

Understand the provided description carefully and assign a label to this image based on what is the "{criterion}" depicted in this image.

Please respond in the following format within five words: "*Answer*". Do not talk about the description and do not respond long sentences. The answer should be within five words.

Again, your job is to understand the description and assign a label to this image based on what is the "{criterion}" shown in this image. 

Your response:
'''

# --- Step 1: Load captions and criterion from JSONL ---
def load_captions(jsonl_path: str) -> List[Tuple[str, str, str]]:
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            image = entry.get("image", "")
            captions = entry.get("captions", {})
            for criterion, caption in captions.items():
                results.append((image, criterion, caption))
    return results


# --- Step 2: Format a batch of captions into the prompt ---
def format_single_prompt(criterion: str, caption: str) -> str:
    return PROMPT_TEMPLATE.format(
        criterion=criterion,
        criterion_specific_caption=f"\"{caption}\""
    )

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
    #print(response)
    marker = "Your response:assistant"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].strip()
    else:
        return "[ERROR]: MARKER NOT FOUND!"  # fallback if marker not found

def clean_criterion(line: str) -> str:
    # Remove leading '*', trim, and remove inline Markdown bold markers
    line = line.strip()
    if line.startswith("*"):
        line = line.lstrip("* ").strip()
        # Remove all instances of '**' (Markdown bold)
        line = re.sub(r"\*\*", "", line)
    return line

# --- Step 4: Run batching and collection ---
# =====This func is for individual processing
def discover_labels_and_save(jsonl_path: str, output_path: str, image_batch_size: int = 50):
    results = load_captions(jsonl_path)
    
    grouped_annotations = defaultdict(dict)  # image_name → {criterion: label}
    batch = {}
    batch_index = 0
    label_count = 0

    for image_name, criterion, caption in results:
        prompt = format_single_prompt(criterion, caption)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = query_llm(chat_prompt)
        label = clean_criterion(response.strip())

        if label:
            # Add label for this criterion
            grouped_annotations[image_name][criterion] = label
            batch.setdefault(image_name, {})[criterion] = label
            label_count += 1

        # Save if we've collected enough unique images
        if len(batch) >= image_batch_size and label_count == 25*image_batch_size:
            to_save = []
            for img, crit_label_dict in batch.items():
                annotations = [{"criterion": c, "label": l} for c, l in crit_label_dict.items()]
                to_save.append({"image": img, "annotations": annotations})

            save_labels_jsonl(to_save, output_path)
            print(f"Saved batch {batch_index} with {len(batch)} images.")
            batch.clear()
            batch_index += 1
            label_count = 0

    # Final save
    if batch:
        to_save = []
        for img, crit_label_dict in batch.items():
            annotations = [{"criterion": c, "label": l} for c, l in crit_label_dict.items()]
            to_save.append({"image": img, "annotations": annotations})

        save_labels_jsonl(to_save, output_path)
        print(f"Saved final batch with {len(batch)} images.")

# --- Save label to JSON ---
def save_labels_jsonl(data: List[dict], output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ======These funcs are for batch processing
def query_llm_batch(prompts: List[str], max_new_tokens: int = 128) -> List[str]:

    torch.cuda.empty_cache()
    # Apply chat template to all prompts
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(chat_prompt)
    
    # Tokenize batch with padding
    inputs = tokenizer(
        chat_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        #max_length=2048  # Adjust based on your needs,
    ).to(model.device)
    
    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode responses
    responses = []
    for i, output in enumerate(outputs):
        # Skip the input tokens to get only the generated part
        input_length = inputs['input_ids'][i].shape[0]
        generated_tokens = output[input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses

def discover_labels_and_save_batched(jsonl_path: str, output_path: str, batch_size: int = 64):
    results = load_captions(jsonl_path)
    
    grouped_annotations = defaultdict(dict)  # image_name → {criterion: label}
    
    # Process in batches
    for i in range(0, len(results), batch_size):
        batch_data = results[i:i + batch_size]
        
        # Prepare prompts for this batch
        prompts = []
        batch_info = []  # Store (image_name, criterion) for each prompt
        
        for image_name, criterion, caption in batch_data:
            prompt = format_single_prompt(criterion, caption)
            prompts.append(prompt)
            batch_info.append((image_name, criterion))
        
        # Process batch
        print(f"Processing batch {i//batch_size + 1}/{(len(results) + batch_size - 1)//batch_size}")
        responses = query_llm_batch(prompts)
        
        # Process responses
        for (image_name, criterion), response in zip(batch_info, responses):
            label = clean_criterion(response.strip())
            if label:
                grouped_annotations[image_name][criterion] = label
    
    #Save all results at the end
    to_save = []
    output_path_test = "./test/testest.json"
    for img, crit_label_dict in grouped_annotations.items():
        annotations = [{"criterion": c, "label": l} for c, l in crit_label_dict.items()]
        to_save.append({"image": img, "annotations": annotations})
    
    save_labels_jsonl(to_save, output_path_test)
    print(f"Saved {len(to_save)} images with annotations.")

# --- Step 5: Run the full pipeline ---
if __name__ == "__main__":
    jsonl_file_path = CAPTION_FILE
    output_path = OUTPUT_DIR
    start_time = time.time()
    #discover_labels_and_save(jsonl_file_path, output_path)
    discover_labels_and_save_batched(jsonl_file_path, output_path, batch_size=64)
    end_time = time.time()

    collapsed_time = end_time - start_time
    print(f"[!!!] Time taken to run the whole process: {collapsed_time:.2f} seconds")
