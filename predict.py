# /predict.py

import torch
from PIL import Image
import os
import argparse
import logging

import config
from src.model import get_orientation_model
from src.utils import get_device, get_data_transforms, setup_logging, load_image_safely

def predict_single_image(model, image_path, device, transforms):
    """Predicts orientation for a single image file."""
    try:
        #image = Image.open(image_path).convert('RGB')
        image = load_image_safely(image_path)
    except FileNotFoundError:
        logging.error(f"File not found: {image_path}")
        return
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return

    input_tensor = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        
    predicted_class = predicted_idx.item()
    result = config.CLASS_MAP[predicted_class]
    
    logging.info(f"-> Image: '{os.path.basename(image_path)}' | Prediction: {result}")


def run_prediction(args):
    """Main prediction routine."""
    setup_logging()
    
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found at {args.model_path}. Please train the model first.")
        return

    device = get_device()
    transforms = get_data_transforms()

    # Load the trained model
    model = get_orientation_model(pretrained=False) # No need to download weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        predict_single_image(model, input_path, device, transforms)
    elif os.path.isdir(input_path):
        logging.info(f"Processing all images in directory: {input_path}")
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            full_path = os.path.join(input_path, image_file)
            predict_single_image(model, full_path, device, transforms)
    else:
        logging.error(f"Input path is not a valid file or directory: {input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict image orientation.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images.')
    parser.add_argument('--model_path', type=str, default=os.path.join(config.MODEL_SAVE_DIR, "orientation_model_v5_epoch_1_vacc_0.6462.pth"), help='Path to the trained model file.')
    
    args = parser.parse_args()
    run_prediction(args)