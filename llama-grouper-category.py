import json
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#-----Load model----
model_path = "/users/adfy064/archive/vision_saved/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# --- Config ---
INPUT_FILE = "llm_outputs_grouper.json"
OUTPUT_DIR = "llama_outputs_test4.json"

# --- Prompt Template (based on Table 18) ---
PROMPT_TEMPLATE = '''
The following is an initial list of "{CRITERION}" categories. These categories might not be at the same semantic granularity level. For example, category 1 could be “cutting vegetables”, while category 2 is simply “cutting”. In this case, category 1 is more specific than category 2.

{MIDDLE_GRAINED_CATEGORY_NAME}

These categories might not be at the same semantic granularity level. For example, category 1 could be “cutting vegetables”, while category 2 is simply “cutting”. In this case, category 1 is more specific than category 2. Your job is to generate a three-level class hierarchy (class taxonomy, where the first level contains more abstract or general coarse-grained classes, the third level contains more specific finegrained classes, and the second level contains intermediate mid-grained classes) of "{CRITERION}" based on the provided list of "{CRITERION}" categories. Follow these steps to generate the hierarchy.

Follow these steps to generate the hierarchy: 

Step 1 - Understand the provided initial list of "{CRITERION}" categories. The following three-level class hierarchy generation steps are all based on the provided initial list. 
Step 2 - Generate a list of abstract or general "{CRITERION}" categories as the first level of the class hierarchy, covering all the concepts present in the initial list. 
Step 3 - Generate a list of middle-grained "{CRITERION}" categories as the second level of the class hierarchy, in which the middle-grained categories are the subcategories of the categories in the first level. The categories in the second-level are more specific than the first level but should still cover and reflect all the concepts present in the initial list. 
Step 4 - Generate a list of more specific fine-grained "{CRITERION}" categories as the third level of the class hierarchy, in which the categories should reflect more specific "{CRITERION}" concepts that you can infer from the initial list. The categories in the third-level are subcategories of the second-level. 
Step 5 - Output the generated three-level class hierarchy as a JSON object where the keys are the level numbers and the values are a flat list of generated categories at each level, structured like: 
{  
“level 1”: [“categories”], 
“level 2”: [“categories”],
“level 3”: [“categories”]
}

Please only output the JSON object in your response and simply use a flat list to store the generated categories at each level.

Your response:
'''

def load_annotations(filepath):
    annotations_per_image = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            image = entry["image"]
            annotations_per_image[image] = {a["criterion"]: a["label"] for a in entry["annotations"]}
    return annotations_per_image

def group_labels_by_criterion(annotations_per_image):
    """
    Groups all labels assigned to each criterion across all images.

    Args:
        annotations_per_image (dict): Output from load_annotations().

    Returns:
        dict: criterion -> set of unique labels
    """
    grouped = defaultdict(set)

    for i, annotations in enumerate(annotations_per_image.values()):
        for criterion, label in annotations.items():
            cleaned_label = label.strip().rstrip('.')
            if cleaned_label and not any(x in cleaned_label.lower() for x in ["no ", "unknown "]):
                grouped[criterion].add(cleaned_label)
    return grouped

def generate_prompt(criterion, labels):
    """
    Formats the full prompt using the template with the criterion and its labels.
    """
    label_list = "\n".join(f'* "{label}"' for label in labels)
    return PROMPT_TEMPLATE.format(CRITERION=criterion, MIDDLE_GRAINED_CATEGORY_NAME=label_list)

def query_llm(prompt):
    """
    Queries the model and returns the decoded output.
    """
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True)

    # To make sure the issue is not prompt is too long
    if tokenized.input_ids.shape[1] > model.config.max_position_embeddings:
        print(f"[WARNING] Prompt too long: {tokenized.input_ids.shape[1]} tokens (limit: {model.config.max_position_embeddings})")

    inputs = tokenized.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16000,           
            temperature=None,               
            do_sample=False,
            #top_k=40,   
            top_p=None,
            #repetition_penalty=1.3,        # Strong anti-repetition
            #no_repeat_ngram_size=3,        # Prevent 3-gram repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(response)
    marker = "Your response:assistant"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].strip()
    else:
        return "[ERROR]: MARKER NOT FOUND!"  # fallback if marker not found


def discover_labels_and_save(grouped_labels):
    
    results = {}
    for i, (criterion, labels) in enumerate(grouped_labels.items()):
        if i >= 5: # Test the first 5 criterion
            break

        print(f"[{i+1}/{len(grouped_labels)}] Processing: {criterion}. Labels count: {len(labels)}")

        prompt = generate_prompt(criterion, labels)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = query_llm(chat_prompt)
        print("Trimmed response: \n", response)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
    
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                json_obj = json.loads(json_str)

                results[criterion] = json_obj
        
            else:
                print(f"[ERROR] No valid JSON bounds found for '{criterion}'")
                results[criterion] = {"error": "No JSON found", "raw": response}
        
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error for '{criterion}': {e}")
            results[criterion] = {"error": str(e), "raw": response}

    with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    annotations_per_image = load_annotations("llm_outputs_grouper.json")
    grouped_labels_data = group_labels_by_criterion(annotations_per_image)

    # Print out the first 2 criterion and the first 10 labels to test
    for i, (criterion, labels) in enumerate(grouped_labels_data.items()):
        if i >= 2:
            break
        print(f"{criterion} ({len(labels)} labels):")
        for j, label in enumerate(sorted(labels)):
            if j >= 10: break
            print(f"  - {label}")
        print()

    discover_labels_and_save(grouped_labels=grouped_labels_data)