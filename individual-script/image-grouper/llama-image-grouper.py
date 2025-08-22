import json
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

#-----Load model----
model_path = "/users/adfy064/archive/vision_saved/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# --- Config ---
INPUT_FILE = "llama_output_refined.json"
OUTPUT_DIR = "./llama-cat-outputs/llm_image_grouper2.json"

# --- Prompt Template ------
PROMPT_TEMPLATE = '''
### Goal Explanation
Hello! I am a machine learning researcher focusing on image categorization based on the aspect of "{CRITERION}" depicted in images.

### Task Instruction
Therefore, I need your assistance in designing a prompt for the Visual Question Answering (VQA) model to help it identify the "{CRITERION}" category in a given image at three different granularity. 

Please help me design and generate this prompt using the following template: "Question: [Generated VQA Prompt Question]. Answer (reply with an abstract, a common, and a specific category name, respectively):". The generated prompt should be simple and straightforward.

### Output Instruction
Please respond with only the generated prompt.

Your response:
'''


def load_criteria(json_path: str="./llama_output_refined.json", use_titles_only=True) -> list[str]:
    """
    Load refined criteria from a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        use_titles_only (bool): If True, extract only the text before the colon (the criterion name).

    Returns:
        List[str]: List of criteria (titles or full text).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_criteria = json.load(f)

    if use_titles_only:
        return [item.split(":")[0].strip() for item in raw_criteria]
    else:
        return raw_criteria
    
def generate_prompt(criterion):
    """
    Formats the full prompt using the template and labels.
    """
    return PROMPT_TEMPLATE.format(CRITERION=criterion)

def query_llm(prompt):
    """
    Queries the model and returns the decoded output.
    """

    model.eval()  # Ensure eval mode
    #torch.cuda.empty_cache()  # Clear cache

    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True)

    inputs = tokenized.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,                       
            do_sample=False,
            temperature=None,   
            #top_k=20,   
            top_p=None,
            #min_p=0,
            #pad_token_id=tokenizer.eos_token_id,
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

def discover_prompt(criteria):
    results = []  # store each Q&A here
    for i, criterion in enumerate(criteria):
        #if i > 2: break
        prompt = PROMPT_TEMPLATE.format(CRITERION=criterion)
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = query_llm(chat_prompt)
        print("Trimmed response: \n", response)

        results.append(response)

        # Write all collected Q&A to JSON file
    with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Responses saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Load criteria
    refined_criteria = load_criteria()
    #print(refined_criteria)

    discover_prompt(refined_criteria)