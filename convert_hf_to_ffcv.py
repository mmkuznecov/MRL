import argparse
import os
from typing import Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import time

from datasets import load_dataset, load_from_disk
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


class HFDatasetWrapper:
    """Wrapper to make HuggingFace dataset compatible with FFCV writer."""
    
    def __init__(self, hf_dataset, image_key: str, label_key: str):
        self.hf_dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Convert numpy.int64 to regular Python int
        if isinstance(idx, np.integer):
            idx = int(idx)
            
        sample = self.hf_dataset[idx]
        
        # Get image
        image = sample[self.image_key]
        
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get label
        label = sample[self.label_key]
        if isinstance(label, list):
            label = label[0]  # Take first label if it's a list
        
        # Convert image to numpy array
        image_array = np.array(image)
        if image_array.ndim == 2:  # Convert grayscale to RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        
        return (image_array, int(label))


def create_ffcv_dataset(
    hf_dataset_name_or_path: str,
    output_path: str,
    max_resolution: int = 512,
    jpeg_quality: int = 90,
    num_workers: int = 8,
    is_from_disk: bool = False,
    split: str = "train"
):
    """
    Convert a Hugging Face dataset to FFCV format.
    
    Args:
        hf_dataset_name_or_path: Hugging Face dataset name or path to saved dataset
        output_path: Path where the FFCV dataset should be saved (should end with .dat)
        max_resolution: Maximum resolution for images
        jpeg_quality: JPEG quality
        num_workers: Number of workers for writing
        is_from_disk: Whether to load dataset from disk
        split: Dataset split to convert
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {hf_dataset_name_or_path}")
    if is_from_disk:
        dataset = load_from_disk(hf_dataset_name_or_path)
    else:
        dataset = load_dataset(hf_dataset_name_or_path, split=split)
    
    print(f"Dataset loaded. Number of samples: {len(dataset)}")
    
    # Check dataset format
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    # Detect field types
    image_key = None
    label_key = None
    
    # Try to find image and label keys
    for key in sample.keys():
        if isinstance(sample[key], Image.Image):
            image_key = key
        elif key == 'label' or key == 'labels':
            label_key = key
    
    if image_key is None:
        raise ValueError("No image field found in dataset. Available keys: " + str(sample.keys()))
    
    if label_key is None:
        # Try alternative naming patterns
        for key in sample.keys():
            if 'class' in key.lower() or 'category' in key.lower():
                label_key = key
                break
    
    if label_key is None:
        raise ValueError("No label field found in dataset. Available keys: " + str(sample.keys()))
    
    print(f"Using image key: '{image_key}', label key: '{label_key}'")
    
    # Create wrapper dataset
    wrapped_dataset = HFDatasetWrapper(dataset, image_key, label_key)
    
    # Define fields for FFCV
    fields = {
        'image': RGBImageField(
            write_mode='jpg',
            max_resolution=max_resolution,
            jpeg_quality=jpeg_quality
        ),
        'label': IntField()
    }
    
    # Create writer and write dataset
    print(f"Writing FFCV dataset to: {output_path}")
    start_time = time.time()
    
    # Correct DatasetWriter initialization
    writer = DatasetWriter(
        output_path,
        fields,
        num_workers=num_workers
    )
    
    # Write the dataset with num_workers as argument to from_indexed_dataset
    writer.from_indexed_dataset(wrapped_dataset)
    
    end_time = time.time()
    print(f"Dataset conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert Hugging Face dataset to FFCV format')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Hugging Face dataset name or path to saved dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for FFCV dataset (should end with .dat)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to convert (default: train)')
    parser.add_argument('--max-resolution', type=int, default=512,
                        help='Maximum resolution for images (default: 512)')
    parser.add_argument('--jpeg-quality', type=int, default=90,
                        help='JPEG quality (default: 90)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for writing (default: 8)')
    parser.add_argument('--from-disk', action='store_true',
                        help='Load dataset from disk instead of downloading')
    
    args = parser.parse_args()
    
    create_ffcv_dataset(
        hf_dataset_name_or_path=args.dataset,
        output_path=args.output,
        max_resolution=args.max_resolution,
        jpeg_quality=args.jpeg_quality,
        num_workers=args.num_workers,
        is_from_disk=args.from_disk,
        split=args.split
    )


if __name__ == "__main__":
    main()
    

'''
# For train split
python convert_hf_to_ffcv.py \
    --dataset "evanarlian/imagenet_1k_resized_256" \
    --output "data/imagenet_train.dat" \
    --split "train" \
    --max-resolution 256 \
    --jpeg-quality 90 \
    --num-workers 8

# For validation split
python convert_hf_to_ffcv.py \
    --dataset "evanarlian/imagenet_1k_resized_256" \
    --output "data/imagenet_val.dat" \
    --split "val" \
    --max-resolution 256 \
    --jpeg-quality 90 \
    --num-workers 8
'''