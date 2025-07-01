# /predict.py

import torch
from PIL import Image
import os
import argparse
import logging
import time # Import the time module

import config
from src.model import get_orientation_model
from src.utils import get_device, get_data_transforms, setup_logging, load_image_safely

def predict_single_image(model, image_path, device, transforms):
    """Predicts orientation for a single image file and logs the time taken."""
    start_time = time.time() # Start timer for this image

    try:
        image = load_image_safely(image_path)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    input_tensor = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        
    predicted_class = predicted_idx.item()
    result = config.CLASS_MAP[predicted_class]
    
    end_time = time.time() # End timer for this image
    duration = end_time - start_time
    
    print(f"-> Image: '{os.path.basename(image_path)}' | Prediction: {result} (Took {duration:.4f} seconds)")


def run_prediction(args):
    """Main prediction routine."""
    setup_logging()
    
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found at {args.model_path}. Please train the model first.")
        return

    device = get_device()
    all_transforms = get_data_transforms()
    transforms = all_transforms['val']

    # Load the trained model
    model = get_orientation_model(config.MODEL_NAME, pretrained=False) # No need to download weights
    
    # Adjust state_dict keys if the model was compiled
    state_dict = torch.load(args.model_path, map_location=device)
    if next(iter(state_dict)).startswith('_orig_mod.'):
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        predict_single_image(model, input_path, device, transforms)
    elif os.path.isdir(input_path):
        print(f"Processing all images in directory: {input_path}")
        total_dir_start_time = time.time() # Start timer for the entire directory
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in directory: {input_path}")
            return

        for image_file in image_files:
            full_path = os.path.join(input_path, image_file)
            predict_single_image(model, full_path, device, transforms)
        
        total_dir_end_time = time.time() # End timer for the entire directory
        total_duration = total_dir_end_time - total_dir_start_time
        print(f"Finished processing directory '{input_path}'. Total time: {total_duration:.4f} seconds for {len(image_files)} images.")
    else:
        print(f"Input path is not a valid file or directory: {input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict image orientation.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images.')
    parser.add_argument('--model_path', type=str, default=os.path.join(config.MODEL_SAVE_DIR, "best_model.pth"), help='Path to the trained model file.')
    
    args = parser.parse_args()
    run_prediction(args)
