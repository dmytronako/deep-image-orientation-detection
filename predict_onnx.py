import torch
import os
import argparse
import logging
import time
import onnxruntime
import numpy as np

import config
import torchvision.transforms as T
from src.utils import setup_logging, load_image_safely

def predict_single_image_onnx(ort_session, image_path, image_transforms):
    """Predicts orientation for a single image file using the ONNX model and logs the time taken."""
    
    start_time = time.time() # Start timer

    try:
        image = load_image_safely(image_path)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    # Apply transformations and convert to NumPy array for ONNX Runtime
    input_tensor = image_transforms(image).unsqueeze(0).cpu().numpy()
    
    # Get input name from the ONNX session
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    
    # Run inference
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Process output: ONNX Runtime returns a list of outputs, we need the first one
    output = torch.from_numpy(ort_outs[0])
    _, predicted_idx = torch.max(output, 1)
        
    predicted_class = predicted_idx.item()
    result = config.CLASS_MAP[predicted_class]
    
    end_time = time.time() # End timer
    duration = end_time - start_time
    
    print(f"-> Image: '{os.path.basename(image_path)}' | Prediction: {result}")


def run_prediction_onnx(args):
    """Main ONNX prediction routine."""
    setup_logging()
    
    if not os.path.exists(args.model_path):
        logging.error(f"ONNX model file not found at {args.model_path}.")
        return

    # Define the same transformations used during validation.
    image_transforms = T.Compose([
        T.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        T.CenterCrop(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ONNX model, explicitly trying to use CUDA
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] # Prioritize CUDA
        ort_session = onnxruntime.InferenceSession(args.model_path, providers=providers)
        logging.info(f"ONNX model loaded from {args.model_path} with providers: {ort_session.get_providers()}")
    except Exception as e:
        logging.error(f"Error loading ONNX model {args.model_path}: {e}")
        return

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        predict_single_image_onnx(ort_session, input_path, image_transforms)
    elif os.path.isdir(input_path):
        print(f"Processing all images in directory: {input_path}")
        total_dir_start_time = time.time() # Start timer for the entire directory
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in directory: {input_path}")
            return

        for image_file in image_files:
            full_path = os.path.join(input_path, image_file)
            predict_single_image_onnx(ort_session, full_path, image_transforms)
        
        total_dir_end_time = time.time() # End timer
        total_duration = total_dir_end_time - total_dir_start_time
        print(f"Finished processing directory '{input_path}'. Total time: {total_duration:.4f} seconds for {len(image_files)} images.")
    else:
        print(f"Input path is not a valid file or directory: {input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict image orientation using an ONNX model.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images.')
    parser.add_argument('--model_path', type=str, default=os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}.onnx"), help='Path to the ONNX model file.')
    
    args = parser.parse_args()
    run_prediction_onnx(args)
