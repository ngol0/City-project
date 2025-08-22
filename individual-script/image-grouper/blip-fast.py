# Fast Batch Processing for Tiny ImageNet
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import os
import json
from datasets import Dataset, load_dataset
import math
from typing import List, Dict, Tuple
from PIL import Image

model_path = "/users/adfy064/archive/vision_saved/blip-instruct"

processor = InstructBlipProcessor.from_pretrained(model_path)
model = InstructBlipForConditionalGeneration.from_pretrained(model_path).to("cuda")
model.eval()

#------ For plain blip ------
# from transformers import AutoProcessor, AutoModelForVision2Seq

# model_path = "/users/adfy064/archive/vision_saved/blip"
# processor = AutoProcessor.from_pretrained(model_path)
# model = AutoModelForVision2Seq.from_pretrained(model_path).to("cuda")
# model.eval()

# ------------ Load question & criteria ---------------------
def load_questions_and_criteria(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    questions = [item["question"] for item in data]
    criteria = [item["criterion"] for item in data]
    
    return questions, criteria

# -------- Load Dataset --------
def load_tiny_net(group: str) -> tuple[Dataset, list[str]]:
    ds = load_dataset("zh-plus/tiny-imagenet")
    img_group = ds[group]
    label_names = img_group.features['label'].names
    return img_group, label_names

# ---------- FAST BATCH PROCESSING -------------------------
def process_image_batch(images: List[Image.Image], questions: List[str], 
                       image_batch_size: int = 4, question_batch_size: int = 3) -> List[List[str]]:
    """
    Process multiple images with multiple questions in efficient batches.
    
    Args:
        images: List of PIL images
        questions: List of questions to ask
        image_batch_size: How many images to process together
        question_batch_size: How many questions to process together
    
    Returns:
        List of lists containing answers for each image
    """
    all_results = []
    
    with torch.no_grad():
        # Process images in batches
        for img_start in range(0, len(images), image_batch_size):
            img_end = min(img_start + image_batch_size, len(images))
            image_batch = images[img_start:img_end]
            
            image_results = [[] for _ in range(len(image_batch))]
            
            # Process questions in batches for each image batch
            for q_start in range(0, len(questions), question_batch_size):
                q_end = min(q_start + question_batch_size, len(questions))
                question_batch = questions[q_start:q_end]
                
                # Create all combinations of images and questions for this batch
                batch_images = []
                batch_questions = []
                batch_indices = []
                
                for img_idx, image in enumerate(image_batch):
                    for q_idx, question in enumerate(question_batch):
                        batch_images.append(image)
                        batch_questions.append(question)
                        batch_indices.append((img_idx, q_start + q_idx))
                
                # Process entire batch at once
                if batch_images:
                    inputs = processor(
                        images=batch_images,
                        text=batch_questions,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to("cuda")
                    
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=3,  # Reduced for speed
                        max_length=64,  # Reduced for speed
                        min_length=1,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                    
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for i, (img_idx, q_idx) in enumerate(batch_indices):
                        answer = generated_texts[i].strip()
                        if ":" in answer:
                            answer = answer.rsplit(":", 1)[-1].strip()
                        
                        # Ensure we have enough slots
                        while len(image_results[img_idx]) <= q_idx:
                            image_results[img_idx].append("")
                        
                        image_results[img_idx][q_idx] = answer
            
            all_results.extend(image_results)
    
    return all_results


# ---------- MAIN FAST PROCESSING FUNCTION -------------------------
def generate_captions_fast_patching(data, label_set: List[str], questions: List[str], 
                                   criteria: List[str], saved_file: str,
                                   memory_patch_size: int = 16,
                                   image_batch_size: int = 4,
                                   question_batch_size: int = 3,
                                   max_samples: int = None) -> List[Dict]:
    """
    Fast processing using multiple batching strategies.
    
    Args:
        data: Dataset to process
        label_set: List of label names
        questions: List of questions
        criteria: List of criteria
        saved_file: Base filename for output
        memory_patch_size: Number of samples to process per memory patch
        image_batch_size: Number of images to process together
        question_batch_size: Number of questions to process together
        use_question_grouping: Group similar questions for efficiency
        max_samples: Maximum number of samples to process
    
    Returns:
        List of results
    """
    all_results = []
    total_samples = len(data) if max_samples is None else min(max_samples, len(data))
    total_patches = math.ceil(total_samples / memory_patch_size)
    
    print(f"Fast processing {total_samples} samples in {total_patches} memory patches")
    print(f"Using image_batch_size={image_batch_size}, question_batch_size={question_batch_size}")
    
    # Process data in memory patches
    for patch_idx in range(total_patches):
        patch_start = patch_idx * memory_patch_size
        patch_end = min(patch_start + memory_patch_size, total_samples)
        
        print(f"\n--- Processing Patch {patch_idx + 1}/{total_patches} ---")
        
        # Collect images and metadata for this patch
        patch_images = []
        patch_metadata = []
        
        for i in range(patch_start, patch_end):
            sample = data[i]
            image = sample['image']
            label_index = sample['label']
            label = label_set[label_index]
            image_filename = f"{label}_{i:04}.jpg"
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                # print(f"Converting {image_filename} from {image.mode} to RGB")
                # image = image.convert('RGB')
                print(f"(x) Skipping grayscale image: {image_filename}")
                continue
            
            patch_images.append(image)
            patch_metadata.append({
                'filename': image_filename,
                'label': label,
                'index': i
            })
        

        # Process all questions together
        batch_answers = process_image_batch(
            patch_images, 
            questions,
            image_batch_size=image_batch_size,
            question_batch_size=question_batch_size
        )
            
        # Create results
        patch_results = []
        for img_idx, (answers, metadata) in enumerate(zip(batch_answers, patch_metadata)):
            result = {
                "image": metadata['filename'],
                "label": metadata['label'],
                "cluster": {}
            }
                
            for ans_idx, answer in enumerate(answers):
                if ans_idx < len(criteria):
                    result["cluster"][criteria[ans_idx]] = answer
                
            patch_results.append(result)
        
        # Save patch results
        save_patch_results(patch_results, patch_idx, saved_file)
        all_results.extend(patch_results)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"Completed patch {patch_idx + 1} with {len(patch_results)} results")
    
    # Merge all patch files
    merge_patch_files(saved_file, total_patches)
    
    print(f"\nFast processing complete! Total results: {len(all_results)}")
    return all_results

def save_patch_results(results: List[Dict], patch_id: int, saved_file: str):
    """Save results from a processing patch immediately."""
    patch_filename = f"{saved_file}_patch_{patch_id:04d}.jsonl"
    
    with open(patch_filename, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

def merge_patch_files(saved_file: str, total_patches: int):
    """Merge all patch files into a single output file."""
    final_filename = f"{saved_file}.jsonl"
    
    with open(final_filename, "w") as outfile:
        for patch_id in range(total_patches):
            patch_filename = f"{saved_file}_patch_{patch_id:04d}.jsonl"
            
            if os.path.exists(patch_filename):
                with open(patch_filename, "r") as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(patch_filename)

# -------- Main Entry Point --------
if __name__ == "__main__":
    print("Loading dataset...")
    tiny_val, label_names = load_tiny_net('valid')
    
    print("Loading questions and criteria...")
    questions, refined_criteria = load_questions_and_criteria("questions_clean.json")
    
    # SPEED-OPTIMIZED Configuration
    json_output_path = "./blip-fast/blip-fast-batch-notinstruct"
    memory_patch_size = 20      # Process more images per memory patch
    image_batch_size = 8        # Process 6 images simultaneously
    question_batch_size = 5     # Process 4 questions simultaneously
    max_samples = None           
    
    print(f"SPEED Configuration:")
    print(f"  - Memory patch size: {memory_patch_size}")
    print(f"  - Image batch size: {image_batch_size}")
    print(f"  - Question batch size: {question_batch_size}")
    print(f"  - Max samples: {max_samples}")
    
    try:
        results = generate_captions_fast_patching(
            data=tiny_val,
            label_set=label_names,
            questions=questions,
            criteria=refined_criteria,
            saved_file=json_output_path,
            memory_patch_size=memory_patch_size,
            image_batch_size=image_batch_size,
            question_batch_size=question_batch_size,
            max_samples=max_samples
        )
        
        print(f"\nSuccess! Processed {len(results)} images with SPEED optimization.")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")