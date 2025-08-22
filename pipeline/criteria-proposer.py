import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
import json
import random
import re
from pathlib import Path
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisionLanguagePipeline:
    """
    Complete pipeline for processing images with MiniCPM and discovering clustering criteria with LLaMA.
    """
    
    def __init__(self, minicpm_model_dir: str, llama_model_dir: str, 
                 device: str = 'cuda', dtype=torch.bfloat16):
        """
        Initialize the complete vision-language pipeline.
        
        Args:
            minicpm_model_dir: Path to MiniCPM model directory
            llama_model_dir: Path to LLaMA model directory  
            device: Device to run models on
            dtype: Data type for models
        """
        self.minicpm_model_dir = minicpm_model_dir
        self.llama_model_dir = llama_model_dir
        self.device = device
        self.dtype = dtype
        
        # Set up proxy if needed
        os.environ["HTTP_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"
        os.environ["HTTPS_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"
        
        # Initialize models
        self.minicpm_model = None
        self.minicpm_tokenizer = None
        self.llama_model = None
        self.llama_tokenizer = None
        
        self._load_models()
    
    def _load_models(self):
        """Load both MiniCPM and LLaMA models."""
        logger.info("Loading MiniCPM model...")
        self.minicpm_model = AutoModel.from_pretrained(
            self.minicpm_model_dir,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        self.minicpm_model = self.minicpm_model.to(device=self.device, dtype=self.dtype)
        self.minicpm_model.eval()
        
        self.minicpm_tokenizer = AutoTokenizer.from_pretrained(
            self.minicpm_model_dir,
            trust_remote_code=True
        )
        
        logger.info("Loading LLaMA model...")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_model_dir)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.llama_model.eval()
        
        logger.info("All models loaded successfully")
    
    @staticmethod
    def load_tiny_imagenet(group: str = "train") -> Tuple[Dataset, List[str]]:
        """Load TinyImageNet dataset."""
        ds = load_dataset("zh-plus/tiny-imagenet")
        img_group = ds[group]
        label_names = img_group.features['label'].names
        return img_group, label_names
    
    def prepare_batch_questions(self, data: Dataset, label_set: List[str],
                               question: str, max_samples: int = None,
                               skip_grayscale: bool = True) -> List[Dict[str, Any]]:
        """Prepare batch questions for MiniCPM processing."""
        questions = []
        processed_count = 0
        
        for i, sample in enumerate(data):
            if max_samples and processed_count >= max_samples:
                break
            
            image = sample['image']
            label_index = sample['label']
            label = label_set[label_index]
            
            # Skip grayscale images if requested
            if skip_grayscale and image.mode != 'RGB':
                logger.info(f"Skipping grayscale image at index {i}")
                continue
            
            # Prepare question in MiniCPM batch format
            question_batch = [{
                "role": "user",
                "content": [image, question]
            }]
            
            questions.append({
                'question': question_batch,
                'image': image,
                'label': label,
                'index': i,
                'filename': f"{label}_{processed_count:04}.jpg"
            })
            
            processed_count += 1
        
        logger.info(f"Prepared {len(questions)} questions for processing")
        return questions
    
    def process_image_batch(self, batch_data: List[Dict[str, Any]],
                           temperature: float = 0.7, sampling: bool = True) -> List[Dict[str, str]]:
        """Process a batch of images with MiniCPM."""
        results = []
        questions = [item['question'] for item in batch_data]
        
        try:
            # Use MiniCPM batch API
            res = self.minicpm_model.chat(
                image=None,
                msgs=questions,
                tokenizer=self.minicpm_tokenizer,
                sampling=sampling,
                temperature=temperature,
                stream=False
            )
            
            # Handle different return formats
            if isinstance(res, tuple):
                responses = res[0] if isinstance(res[0], list) else [res[0]]
            elif isinstance(res, list):
                responses = res
            else:
                responses = [res]
            
            # Ensure correct number of responses
            if len(responses) != len(batch_data):
                logger.warning(f"Expected {len(batch_data)} responses, got {len(responses)}")
                while len(responses) < len(batch_data):
                    responses.append("Error: No response received")
            
            # Combine results with metadata
            for batch_item, response in zip(batch_data, responses):
                results.append({
                    "image": batch_item['filename'],
                    "label": batch_item['label'],
                    "description": response,
                    "original_index": batch_item['index']
                })
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            results = self._process_images_individually(batch_data, temperature, sampling)
        
        return results
    
    def _process_images_individually(self, batch_data: List[Dict[str, Any]],
                                   temperature: float, sampling: bool) -> List[Dict[str, str]]:
        """Fallback individual processing for images."""
        results = []
        
        for item in batch_data:
            try:
                question = item['question']
                image = question[0]['content'][0]
                text_question = question[0]['content'][1]
                
                msgs = [{'role': 'user', 'content': text_question}]
                
                res, context, _ = self.minicpm_model.chat(
                    image=image,
                    msgs=msgs,
                    context=None,
                    tokenizer=self.minicpm_tokenizer,
                    sampling=sampling,
                    temperature=temperature
                )
                
                results.append({
                    "image": item['filename'],
                    "label": item['label'],
                    "description": res,
                    "original_index": item['index']
                })
                
            except Exception as e:
                logger.error(f"Error processing individual item {item['index']}: {e}")
                results.append({
                    "image": item['filename'],
                    "label": item['label'],
                    "description": f"Error: {str(e)}",
                    "original_index": item['index']
                })
        
        return results
    
    def save_images_and_descriptions(self, batch_data: List[Dict[str, Any]],
                                   results: List[Dict[str, str]], output_dir: str):
        """Save processed images and their descriptions."""
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for batch_item, result in zip(batch_data, results):
            image_path = os.path.join(images_dir, result['image'])
            batch_item['image'].save(image_path)
        
        logger.info(f"Saved {len(results)} images to {images_dir}")
    
    def process_images_to_descriptions(self, output_dir: str = "./pipeline_outputs",
                                     question: str = 'Describe the following image in detail.',
                                     max_samples: int = 400, batch_size: int = 4,
                                     dataset_split: str = "train") -> str:
        """
        Step 1: Process images and generate descriptions.
        Returns path to the descriptions JSON file.
        """
        logger.info("=== Step 1: Processing Images with MiniCPM ===")
        
        # Load dataset
        data, label_names = self.load_tiny_imagenet(dataset_split)
        max_samples = len(data) if max_samples is None else min(max_samples, len(data))
        
        # Prepare questions
        prepared_data = self.prepare_batch_questions(
            data, label_names, question, max_samples
        )
        
        # Process in batches
        all_results = []
        for i in range(0, len(prepared_data), batch_size):
            batch = prepared_data[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(prepared_data) + batch_size - 1)//batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")
            
            batch_results = self.process_image_batch(batch)
            all_results.extend(batch_results)
            
            # Save images for this batch
            #self.save_images_and_descriptions(batch, batch_results, output_dir)
            
            logger.info(f"Batch {batch_num} complete. Total processed: {len(all_results)}")
        
        # Save descriptions to JSON
        descriptions_path = os.path.join(output_dir, "descriptions.json")
        self.save_to_json(descriptions_path, all_results)
        
        logger.info(f"Step 1 complete! Processed {len(all_results)} images.")
        return descriptions_path
    
    def load_descriptions_from_json(self, json_path: str) -> List[str]:
        """Load descriptions from JSON file generated by image processing step."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [item['description'] for item in data]
    
    def format_llama_prompt(self, captions: List[str]) -> str:
        """Format captions into LLaMA prompt template."""
        prompt_template = '''
The following are the result of captioning a set of images:
{captions}

I am a machine learning researcher trying to figure out the potential clustering or grouping criteria that exist in these images. So I can better understand my data and group them into different clusters based on different criteria.

Come up with ten distinct clustering criteria that exist in this set of images.

Please write a list of clustering criteria (separated by bullet points "*").

Again I want to figure out what are the potential clustering/grouping criteria that I can use to group these images into different clusters. List ten clustering or grouping criteria that often exist in this set of images based on the captioning results. Answer with a list (separated by bullet points "*").

Your response:
'''
        caption_block = "\n".join([f"Image {i+1}: \"{caption}\"" for i, caption in enumerate(captions)])
        return prompt_template.format(captions=caption_block)
    
    def build_chat_prompt(self, captions: List[str]) -> str:
        """Build chat prompt for LLaMA using messages format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.format_llama_prompt(captions)}
        ]
        return self.llama_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def query_llama(self, prompt: str, max_new_tokens: int = 4000) -> str:
        """Query LLaMA model for clustering criteria."""
        inputs = self.llama_tokenizer(prompt, return_tensors="pt", truncation=True).to(self.llama_model.device)
        
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                eos_token_id=self.llama_tokenizer.eos_token_id,
            )
        
        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after the marker
        marker = "Your response:assistant"
        idx = response.find(marker)
        if idx != -1:
            return response[idx + len(marker):].strip()
        else:
            # Fallback: try to extract everything after "Your response:"
            marker_fallback = "Your response:"
            idx = response.find(marker_fallback)
            if idx != -1:
                return response[idx + len(marker_fallback):].strip()
            return response.strip()  # Return full response as fallback
    
    @staticmethod
    def clean_criterion(line: str) -> str:
        """Clean and format criterion line."""
        line = line.strip()
        if line.startswith("*"):
            line = line.lstrip("* ").strip()
            # Remove Markdown bold markers
            line = re.sub(r"\*\*", "", line)
        return line
    
    def discover_clustering_criteria(self, descriptions_path: str, batch_size: int = 400,
                                   output_dir: str = "./pipeline_outputs") -> List[str]:
        """
        Step 2: Discover clustering criteria using LLaMA.
        """
        logger.info("=== Step 2: Discovering Clustering Criteria with LLaMA ===")
        
        # Load descriptions
        captions = self.load_descriptions_from_json(descriptions_path)
        random.shuffle(captions)
        
        all_criteria = set()
        
        # Process in batches
        for i in range(0, len(captions), batch_size):
            batch = captions[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(captions) + batch_size - 1) // batch_size
            
            logger.info(f"Processing criteria batch {batch_num}/{total_batches} ({len(batch)} descriptions)")
            
            prompt = self.build_chat_prompt(batch)
            response = self.query_llama(prompt)
            
            logger.info(f"Batch {batch_num} Response:\n{response}\n")
            
            # Extract criteria from response
            for line in response.split("\n"):
                cleaned = self.clean_criterion(line)
                if cleaned and len(cleaned) > 3:  # Filter out very short lines
                    all_criteria.add(cleaned)
        
        criteria_list = sorted(all_criteria)
        
        # Save criteria
        criteria_path = os.path.join(output_dir, "clustering_criteria.json")
        self.save_to_json(criteria_path, criteria_list)
        
        logger.info(f"Step 2 complete! Discovered {len(criteria_list)} unique criteria.")
        return criteria_list
    
    @staticmethod
    def save_to_json(path: str, data: Any):
        """Save data to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {path}")
    
    def run_complete_pipeline(self, output_dir: str = "./pipeline_outputs",
                            image_question: str = 'Describe the following image in detail.',
                            max_samples: int = 400, image_batch_size: int = 4,
                            criteria_batch_size: int = 400, dataset_split: str = "valid",
                            run_llama_only: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline: image processing -> criteria discovery.
        
        Returns:
            Dictionary with paths to generated files and summary statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_output_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(timestamped_output_dir, exist_ok=True)
        
        logger.info(f"=== Starting Complete Vision-Language Pipeline ===")
        logger.info(f"Output directory: {timestamped_output_dir}")
        logger.info(f"Max samples: {max_samples}")
        logger.info(f"Image batch size: {image_batch_size}")
        logger.info(f"Criteria batch size: {criteria_batch_size}")
        
        try:
            if run_llama_only: # File name after step 1
                descriptions_path = "descriptions.json"
            else:
                #Step 1: Process images to descriptions
                descriptions_path = self.process_images_to_descriptions(
                    output_dir=timestamped_output_dir,
                    question=image_question,
                    max_samples=max_samples,
                    batch_size=image_batch_size,
                    dataset_split=dataset_split
                )

            # Step 2: Discover clustering criteria
            criteria_list = self.discover_clustering_criteria(
                descriptions_path=descriptions_path,
                batch_size=criteria_batch_size,
                output_dir=timestamped_output_dir
            )
            
            # Create summary
            summary = {
                "pipeline_completed": True,
                "timestamp": timestamp,
                "output_directory": timestamped_output_dir,
                "total_images_processed": max_samples,
                "total_criteria_discovered": len(criteria_list),
                "descriptions_file": descriptions_path,
                "criteria_file": os.path.join(timestamped_output_dir, "clustering_criteria.json"),
                "images_directory": os.path.join(timestamped_output_dir, "images"),
                "criteria_preview": criteria_list[:10] if len(criteria_list) > 10 else criteria_list
            }
            
            # Save pipeline summary
            summary_path = os.path.join(timestamped_output_dir, "pipeline_summary.json")
            self.save_to_json(summary_path, summary)
            
            logger.info("=== Pipeline Complete! ===")
            logger.info(f"Images processed: {summary['total_images_processed']}")
            logger.info(f"Criteria discovered: {summary['total_criteria_discovered']}")
            logger.info(f"Results saved to: {timestamped_output_dir}")
            
            # Print discovered criteria
            logger.info("\n=== Discovered Clustering Criteria ===")
            for i, criterion in enumerate(criteria_list, 1):
                logger.info(f"{i:2d}. {criterion}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e


def main():
    """Main execution function."""
    # Configuration
    minicpm_model_dir = "/users/adfy064/archive/vision_saved/minicpm26"
    llama_model_dir = "/users/adfy064/archive/vision_saved/llama3"
    
    # Initialize pipeline
    pipeline = VisionLanguagePipeline(
        minicpm_model_dir=minicpm_model_dir,
        llama_model_dir=llama_model_dir
    )

    # If Step 2 fails while running, set this to True to only run Step 2 after Step 1 is done
    run_step_2_only = False
    
    # Run complete pipeline
    summary = pipeline.run_complete_pipeline(
        output_dir="./complete_pipeline_outputs",
        image_question='Describe the following image in detail.',
        max_samples=None,      # If None -> whole dataset, otherwise use certain samples
        image_batch_size=50,   # Number of images to process in batch with MiniCPM
        criteria_batch_size=400,
        dataset_split="valid",
        run_llama_only=run_step_2_only
    )
    
    print(f"\nPipeline completed successfully!")
    print(f"Check results in: {summary['output_directory']}")


if __name__ == "__main__":
    main()