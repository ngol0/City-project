import torch
from transformers import AutoModel, AutoTokenizer
import os
import json
from torchvision import datasets
from PIL import Image
from datasets import Dataset, load_dataset


os.environ["HTTP_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"
os.environ["HTTPS_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"

#---------Load MiniCPM-----------
model_dir = "/users/adfy064/archive/vision_saved/minicpm-v2"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='cuda', dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model.eval()

# -----Load Dataset
def load_tiny_net(group: str) -> tuple[Dataset, list[str]]:
    ds = load_dataset("zh-plus/tiny-imagenet")
    img_group = ds[group]
    label_names = img_group.features['label'].names

    return img_group, label_names

# --------
def minicpm_for_natural_images(data, label, images_dir, question) -> list[dict[str, str]]:
    results = []
    for i, sample in enumerate(data):
        #if i > 2: break
        image = sample['image']
        label_index = sample['label']
        label = label[label_index]
        image_filename = f"{label}_{i:04}.jpg"
        image_path = os.path.join(images_dir, image_filename)

        # Check if the image is in RGB mode (skip grayscale images)
        if image.mode != 'RGB':
            print(f"(x) Skipping grayscale image: {image_filename}")
            continue

        # Save the image
        image.save(image_path)

        # Generate description
        msgs = [{'role': 'user', 'content': question}]
        res, context, _ = model.chat(image=image,
                                msgs=msgs,
                                context=None,
                                tokenizer=tokenizer,
                                sampling=True,
                                temperature=0.7)

        # Save the description
        results.append({
            "image": image_filename,
            "description": res
        })
        
        if i % 100 == 0:
            print(f"[{i}] Saved {image_filename} + description")

    return results

def save_to_json(output):
    with open("descriptions_med.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    tiny_train, label = load_tiny_net("train")

    #--Config path and dir
    output_dir = "./minicpm_outputs"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "descriptions.csv")

    question = 'Describe the following image in detail.'

    output = minicpm_for_natural_images(tiny_train, label, images_dir, question)
    save_to_json(output)