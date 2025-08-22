import torch
from transformers import AutoModel, AutoTokenizer
import os
import json
from torchvision import datasets
from PIL import Image
from datasets import Dataset, load_dataset
from functools import partial

# -------- Proxy and Device Setup --------
os.environ["HTTP_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"
os.environ["HTTPS_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"

# -------- Load MiniCPM Model --------
model_dir = "/users/adfy064/archive/vision_saved/minicpm-v2"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model.eval()

# -------- Load Dataset --------
def load_tiny_net(group: str) -> tuple[Dataset, list[str]]:
    ds = load_dataset("zh-plus/tiny-imagenet")
    img_group = ds[group]
    label_names = img_group.features['label'].names
    return img_group, label_names

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


# -------- Criterion-Specific Captioning --------
def generate_captions_by_criterion(data, label_set, images_dir, criteria, saved_file) -> list[dict]:
    results = []

    for i, sample in enumerate(data):
        #if i > 2: break
        image = sample['image']
        label_index = sample['label']
        label = label_set[label_index]
        image_filename = f"{label}_{i:04}.jpg"
        image_path = os.path.join(images_dir, image_filename)

        if image.mode != 'RGB':
            print(f"(x) Skipping grayscale image: {image_filename}")
            continue

        #image.save(image_path)  # Optional: save image

        image_results = {
            "image": image_filename,
            "captions": {}
        }

        for criterion in criteria:
            #print("Criterion: ", criterion)
            prompt = f"""Analyze the image focusing specifically on the "{criterion}". Provide a detailed description of the "{criterion}" depicted in the image. Highlight key elements and interactions relevant to the "{criterion}" that enhance the understanding of the scene."""
            
            msgs = [{'role': 'user', 'content': prompt}]
            res, context, _ = model.chat(image=image,
                                         msgs=msgs,
                                         context=None,
                                         tokenizer=tokenizer,
                                         sampling=True,
                                         temperature=0.7)
            image_results["captions"][criterion] = res

        results.append(image_results)

        if i % 100 == 0:
            print(f"[{i}] Processed: {image_filename}")

        if len(results) % 100 == 0:
            print(f"[{0 + len(results)}] Saved {image_filename} + description")
            with open(f"{saved_file}.jsonl", "a") as f:
                for entry in results[-100:]:  # only save the last 100
                    f.write(json.dumps(entry) + "\n")

    return results

def generate_captions_by_criterion_slicing(start_index, end_index, data, label_set, images_dir, criteria, saved_file) -> list[dict]:
    results = []

    for idx in range(start_index, end_index):
        image = data[idx]['image']
        label_index = data[idx]['label']
        label = label_set[label_index]
        image_filename = f"{label}_{idx:04}.jpg"
        image_path = os.path.join(images_dir, image_filename)

        if image.mode != 'RGB':
            print(f"(x) Skipping grayscale image: {image_filename}")
            continue

        #image.save(image_path)  # Optional: save image

        image_results = {
            "image": image_filename,
            "captions": {}
        }

        for criterion in criteria:
            #print("Criterion: ", criterion)
            prompt = f"""Analyze the image focusing specifically on the "{criterion}". Provide a detailed description of the "{criterion}" depicted in the image. Highlight key elements and interactions relevant to the "{criterion}" that enhance the understanding of the scene."""
            
            msgs = [{'role': 'user', 'content': prompt}]
            res, context, _ = model.chat(image=image,
                                         msgs=msgs,
                                         context=None,
                                         tokenizer=tokenizer,
                                         sampling=True,
                                         temperature=0.7)
            image_results["captions"][criterion] = res

        results.append(image_results)

        if idx % 100 == 0:
            print(f"[{idx}] Processed: {image_filename}")

        if len(results) % 100 == 0:
            print(f"[{3801 + len(results)}] Saved {image_filename} + description")
            with open(f"{saved_file}.jsonl", "a") as f:
                for entry in results[-100:]:  # only save the last 100
                    f.write(json.dumps(entry) + "\n")

    return results
# -------- Save Output to JSON --------
# def save_to_json(path: str, output):
#     with open(path, "w") as f:
#         json.dump(output, f, indent=4)

def save_rest(file_name: str, results: list[dict]):
    """
    Save the remaining unsaved entries (i.e., len(results) % 100) at the end of the run.

    Args:
        file_name (str): Base file name to write to (without .jsonl).
        results (list): List of all generated image-caption results.
    """
    remainder = len(results) % 100
    if remainder == 0:
        print("[✓] No remaining results to save.")
        return

    print(f"[{len(results) - remainder}] Saving the final {remainder} results to {file_name}.jsonl")

    with open(f"{file_name}.jsonl", "a") as f:
        for entry in results[-remainder:]:
            f.write(json.dumps(entry) + "\n")

# -------- Main Entry Point --------
if __name__ == "__main__":
    # Load dataset
    tiny_val, label_names = load_tiny_net('valid')

    # Output config for images
    output_dir = "./minicpm_outputs"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # for save file
    json_output_path = "criterion_captions_test-3801"

    # Define refined criteria
    refined_criteria = load_criteria()

    SLICE_INDEX = True

    if SLICE_INDEX:
        start = 3801 # change this if needed
        end = 7600 # change this if needed
        result = generate_captions_by_criterion_slicing(
            start, end, 
            tiny_val, label_names,
            images_dir,
            refined_criteria,
            json_output_path)
    else:
        result = generate_captions_by_criterion(
            data=tiny_val,
            label_set=label_names,
            images_dir=images_dir,
            criteria=refined_criteria,
            saved_file=json_output_path
        )

    #save_to_json(json_output_path, result)
    save_rest(json_output_path, result)
