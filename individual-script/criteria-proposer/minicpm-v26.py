import torch
from transformers import AutoModel, AutoTokenizer
import os
import json
from torchvision import datasets
from PIL import Image
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MiniCPMBatchProcessor:
    def __init__(self, model_dir: str, device: str = 'cuda', dtype=torch.bfloat16):
        """
        Initialize the MiniCPM batch processor.
        
        Args:
            model_dir: Path to the saved MiniCPM model directory
            device: Device to run the model on ('cuda' or 'cpu')
            dtype: Data type for the model
        """
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        
        # Set up proxy if needed
        os.environ["HTTP_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"
        os.environ["HTTPS_PROXY"] = "http://hpc-proxy00.city.ac.uk:3128"
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the MiniCPM model and tokenizer."""
        logger.info(f"Loading MiniCPM model from {self.model_dir}")
        
        self.model = AutoModel.from_pretrained(
            self.model_dir, 
            trust_remote_code=True, 
            torch_dtype=self.dtype
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, 
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    @staticmethod
    def load_tiny_imagenet(group: str = "train") -> Tuple[Dataset, List[str]]:
        """
        Load TinyImageNet dataset.
        
        Args:
            group: Dataset split to load ('train', 'valid', or 'test')
            
        Returns:
            Tuple of (dataset, label_names)
        """
        ds = load_dataset("zh-plus/tiny-imagenet")
        img_group = ds[group]
        label_names = img_group.features['label'].names
        
        return img_group, label_names
    
    def prepare_batch_questions(self, data: Dataset, label_set: List[str], 
                               question: str, max_samples: int = None, 
                               skip_grayscale: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Prepare batch questions in the format expected by MiniCPM.
        
        Args:
            data: Dataset containing images and labels
            label_set: List of label names
            question: Question to ask about each image
            max_samples: Maximum number of samples to process (None for all)
            skip_grayscale: Whether to skip grayscale images
            
        Returns:
            List of questions in batch format
        """
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
            
            # Prepare question in the required format
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
    
    def process_batch(self, batch_data: List[Dict[str, Any]], 
                     temperature: float = 0.7, sampling: bool = True) -> List[Dict[str, str]]:
        """
        Process a batch of questions using MiniCPM's batch API.
        
        Args:
            batch_data: List of prepared question data
            temperature: Sampling temperature
            sampling: Whether to use sampling
            
        Returns:
            List of results with descriptions
        """
        results = []
        
        # Extract questions for batch processing
        questions = [item['question'] for item in batch_data]
        
        try:
            # Process batch using true batch API
            responses = self._process_questions_batch(questions, temperature, sampling)
            
            # Ensure we have the right number of responses
            if len(responses) != len(batch_data):
                logger.warning(f"Expected {len(batch_data)} responses, got {len(responses)}")
                # Pad with error messages if needed
                while len(responses) < len(batch_data):
                    responses.append("Error: No response received")
            
            # Combine results with metadata
            for i, (batch_item, response) in enumerate(zip(batch_data, responses)):
                results.append({
                    "image": batch_item['filename'],
                    "label": batch_item['label'],
                    "description": response,
                    "original_index": batch_item['index']
                })
                
        except Exception as e:
            logger.error(f"Batch processing failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            results = self._process_individually(batch_data, temperature, sampling)
            
        return results
    
    def _process_questions_batch(self, questions: List[List[Dict[str, Any]]], 
                                temperature: float, sampling: bool) -> List[str]:
        """
        Process questions in true batch mode using MiniCPM's batch API.
        
        Args:
            questions: List of question batches in MiniCPM format
            temperature: Sampling temperature
            sampling: Whether to use sampling
            
        Returns:
            List of response strings
        """
        try:
            # Use the batch processing format: msgs=[msgs, msgs] with image=None
            res = self.model.chat(
                image=None,
                msgs=questions,  # Pass the list of message batches directly
                tokenizer=self.tokenizer,
                sampling=sampling,
                temperature=temperature,
                stream=False
            )
            
            # Handle different possible return formats
            if isinstance(res, tuple):
                # If it returns (responses, context, _), take the first element
                responses = res[0] if isinstance(res[0], list) else [res[0]]
            elif isinstance(res, list):
                # If it directly returns a list of responses
                responses = res
            else:
                # If it returns a single response for some reason
                responses = [res]
            
            logger.info(f"Successfully processed batch of {len(questions)} questions")
            return responses
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise e
    
    def _process_individually(self, batch_data: List[Dict[str, Any]], 
                             temperature: float, sampling: bool) -> List[Dict[str, str]]:
        """Fallback method to process questions individually."""
        results = []
        
        for item in batch_data:
            try:
                question = item['question']
                image = question[0]['content'][0]
                text_question = question[0]['content'][1]
                
                msgs = [{'role': 'user', 'content': text_question}]
                
                res, context, _ = self.model.chat(
                    image=image,
                    msgs=msgs,
                    context=None,
                    tokenizer=self.tokenizer,
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
    
    def save_images_and_results(self, batch_data: List[Dict[str, Any]], 
                               results: List[Dict[str, str]], 
                               output_dir: str) -> None:
        """
        Save processed images and results.
        
        Args:
            batch_data: Original batch data with images
            results: Processing results
            output_dir: Output directory
        """
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for batch_item, result in zip(batch_data, results):
            # Save image
            image_path = os.path.join(images_dir, result['image'])
            batch_item['image'].save(image_path)
            
        logger.info(f"Saved {len(results)} images to {images_dir}")
    
    @staticmethod
    def save_to_json(path: str, data: Any) -> None:
        """Save data to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Results saved to {path}")
    
    def process_dataset(self, output_dir: str = "./minicpm_outputs", 
                       json_filename: str = "descriptions_batch.json",
                       question: str = 'Describe the following image in detail.',
                       max_samples: int = 6, batch_size: int = 4,
                       dataset_split: str = "valid") -> None:
        """
        Main method to process the entire dataset.
        
        Args:
            output_dir: Output directory for results
            json_filename: Name of the JSON output file
            question: Question to ask about images
            max_samples: Maximum number of samples to process
            batch_size: Number of samples to process in each batch
            dataset_split: Dataset split to use
        """
        logger.info("Starting dataset processing...")
        
        # Load dataset
        data, label_names = self.load_tiny_imagenet(dataset_split)
        
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
            
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Save images for this batch
            self.save_images_and_results(batch, batch_results, output_dir)
            
            # Log progress
            logger.info(f"Batch {batch_num} complete. Total processed: {len(all_results)}/{len(prepared_data)}")
            
            if batch_num % 5 == 0:  # Log every 5 batches
                logger.info(f"Progress update: {len(all_results)} samples processed so far...")
        
        # Save all results to JSON
        json_path = os.path.join(output_dir, json_filename)
        self.save_to_json(json_path, all_results)
        
        logger.info(f"Processing complete! Processed {len(all_results)} samples.")


def main():
    """Main execution function."""
    # Configuration
    model_dir = "/users/adfy064/archive/vision_saved/minicpm26"
    output_dir = "./minicpm_outputs"
    
    # Initialize processor
    processor = MiniCPMBatchProcessor(model_dir)
    
    # Process dataset
    processor.process_dataset(
        output_dir=output_dir,
        json_filename="descriptions_batch.json",
        question='Describe the following image in detail.',
        max_samples=20,  # Adjust as needed
        batch_size=10    # Adjust based on your GPU memory
    )


if __name__ == "__main__":
    main()