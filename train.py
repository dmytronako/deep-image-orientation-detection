# /train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import argparse
import logging
import shutil
import time

import config
from src.caching import cache_dataset
from src.dataset import ImageOrientationDataset, ImageOrientationDatasetFromCache
from src.model import get_orientation_model
from src.utils import get_device, setup_logging, get_data_transforms

def train(args):
    """Main training routine."""
    setup_logging()
    training_start_time = time.time()

    logging.info("=================================================")
    logging.info("      STARTING MODEL TRAINING SCRIPT")
    logging.info("=================================================")
    logging.info("Configuration:")
    logging.info(f"  - Using Cache: {config.USE_CACHE}")
    if config.USE_CACHE:
        logging.info(f"  - Cache Directory: {config.CACHE_DIR}")
        logging.info(f"  - Force Rebuild Cache: {args.force_rebuild_cache}")
    logging.info(f"  - Source Data Directory: {args.data_dir}")
    logging.info(f"  - Model Save Directory: {args.model_dir}")
    logging.info(f"  - Number of Epochs: {args.epochs}")
    logging.info(f"  - Batch Size: {args.batch_size}")
    logging.info(f"  - Learning Rate: {args.lr}")
    logging.info(f"  - Dataloader Workers: {args.workers}")
    
    # Ensure model save directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    device = get_device()
    transforms = get_data_transforms()

    # --- Dataset and Dataloaders ---
    logging.info("\n--- Initializing Dataset and Dataloaders ---")

    try:
        if config.USE_CACHE:
            # 1. Trigger the caching process
            cache_dataset(force_rebuild=args.force_rebuild_cache)
            # 2. Use the dataset that reads from the cache
            full_dataset = ImageOrientationDatasetFromCache(
                cache_dir=config.CACHE_DIR,
                transform=transforms
            )
            logging.info(f"Successfully loaded dataset from CACHE ({len(full_dataset)} images).")
        else:
            # Use the original on-the-fly dataset
            logging.info("Using ON-THE-FLY image processing (caching is disabled).")
            full_dataset = ImageOrientationDataset(
                upright_dir=args.data_dir,
                transform=transforms
            )
            logging.info(f"Successfully loaded dataset for on-the-fly processing.")

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if config.USE_CACHE:
        logging.info(f"Total cached dataset size: {len(full_dataset)}")
    else:
        logging.info(f"Dataset found {len(full_dataset.image_files)} original image files.")
        logging.info(f"Total dataset size (with 4 rotations): {len(full_dataset)}")
    
    logging.info(f"Splitting into Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    logging.info("Dataloaders created successfully.")

    # --- Model, Loss, Optimizer ---
    logging.info("\n--- Setting up Model ---") ### <<< SECTION HEADER
    model = get_orientation_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    logging.info(f"Using pre-trained ResNet18 model. Final layer is trainable.")

    # --- Training Loop ---
    best_val_acc = 0.0
    best_model_path = ""

    logging.info("\n--- Starting Training Loop ---") ### <<< SECTION HEADER
    for epoch in range(args.epochs):
        epoch_start_time = time.time() ### <<< START EPOCH TIMER

        # --- Training Phase ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.float() / len(val_dataset)
        
        epoch_duration = time.time() - epoch_start_time ### <<< CALCULATE EPOCH DURATION

        ### <<< MODIFIED: Added epoch duration to the log line
        logging.info(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        # --- MODEL SAVING LOGIC ---
        current_acc = val_epoch_acc.item()
        if current_acc > best_val_acc:
            if os.path.exists(best_model_path):
                logging.debug(f"Removing old best model: {os.path.basename(best_model_path)}")
                os.remove(best_model_path)

            best_val_acc = current_acc
            unique_filename = f"{config.MODEL_NAME}_epoch_{epoch+1}_vacc_{best_val_acc:.4f}.pth"
            unique_save_path = os.path.join(args.model_dir, unique_filename)
            static_save_path = os.path.join(args.model_dir, "best_model.pth")
            
            torch.save(model.state_dict(), unique_save_path)
            shutil.copy(unique_save_path, static_save_path)
            best_model_path = unique_save_path

            logging.info(f"  -> 🎉 New best model saved! Val Acc: {best_val_acc:.4f}")


    ### <<< START: FINAL SUMMARY BLOCK
    total_duration = time.time() - training_start_time
    total_minutes = total_duration / 60
    logging.info("\n=================================================")
    logging.info("              TRAINING COMPLETE")
    logging.info("=================================================")
    logging.info(f"Total Training Time: {total_duration:.2f} seconds ({total_minutes:.2f} minutes)")
    if best_model_path:
        logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logging.info(f"Final best model saved as: {os.path.basename(best_model_path)}")
        logging.info(f"A copy is also available as: best_model.pth")
    else:
        logging.warning("No model was saved as validation accuracy did not improve from its initial state.")
    logging.info("=================================================")
    ### <<< END: FINAL SUMMARY BLOCK


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an image orientation detection model.")
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Directory with upright images.')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_SAVE_DIR, help='Directory to save trained models.')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate.')
    ### <<< ADDED: Argument for number of workers
    parser.add_argument('--workers', type=int, default=config.NUM_WORKERS, help='Number of data loading workers.')
    parser.add_argument('--force-rebuild-cache', action='store_true', help='If set, clears and rebuilds the image cache.')
    
    args = parser.parse_args()
    train(args)