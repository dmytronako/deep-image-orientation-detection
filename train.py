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

import torch.amp as amp # Updated for modern AMP API
import config
from src.caching import cache_dataset
from src.dataset import ImageOrientationDataset, ImageOrientationDatasetFromCache
from src.model import get_orientation_model
from src.utils import get_device, setup_logging, get_data_transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import draw_bounding_boxes

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
    
    writer = SummaryWriter(f"runs/{config.MODEL_NAME}")

    # Ensure model save directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    device = get_device()
    
    # Determine if pin_memory should be used
    pin_memory_enabled = device.type == 'cuda'
    if pin_memory_enabled:
        logging.info("CUDA detected, pin_memory will be enabled for DataLoaders.")
    else:
        logging.info("CUDA not detected, pin_memory will be disabled.")


    # --- Dataset and Dataloaders ---
    logging.info("\n--- Initializing Dataset and Dataloaders ---") 
    data_transforms = get_data_transforms()

    try:
        if config.USE_CACHE:
            # 1. Trigger the caching process
            cache_dataset(force_rebuild=args.force_rebuild_cache)
            # 2. Use the dataset that reads from the cache, but WITHOUT a transform initially
            full_dataset = ImageOrientationDatasetFromCache(
                cache_dir=config.CACHE_DIR,
                transform=None  # IMPORTANT: Apply transform only after splitting
            )
            logging.info(f"Successfully loaded dataset from CACHE ({len(full_dataset)} images).")
        else:
            
            logging.info("Using ON-THE-FLY image processing (caching is disabled).")
            full_dataset = ImageOrientationDataset(
                upright_dir=args.data_dir,
                transform=None # IMPORTANT: Apply transform after splitting
            )
            logging.info(f"Successfully loaded dataset for on-the-fly processing.")

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    # Split the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_subset.dataset.transform = data_transforms['train']
    val_subset.dataset.transform = data_transforms['val']

    if config.USE_CACHE:
        logging.info(f"Total cached dataset size: {len(full_dataset)}")
    else:
        logging.info(f"Dataset found {len(full_dataset.image_files)} original image files.")
        logging.info(f"Total dataset size (with 4 rotations): {len(full_dataset)}")
    
    logging.info(f"Splitting into Training: {len(train_subset)} samples, Validation: {len(val_subset)} samples.")
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin_memory_enabled)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory_enabled)
    logging.info("Dataloaders created successfully.")

    # --- Model, Loss, Optimizer ---
    logging.info("\n--- Setting up Model ---")
    
    # Store the original model instance
    original_model = get_orientation_model().to(device)

    # This will be the model instance used for training/inference during the loop
    # It might be the original_model itself, or its compiled version.
    model_for_training = original_model 

    # Compile the model for performance if PyTorch 2.0+ is used
    if hasattr(torch, 'compile'):
        logging.info("PyTorch 2.0+ detected. Compiling the model for performance...")
        model_for_training = torch.compile(original_model) # Assign the compiled wrapper
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Add label_smoothing
    
    # Initialize optimizer with parameters of the model *used for training* (which might be compiled)
    optimizer = optim.AdamW(model_for_training.parameters(), lr=args.lr, weight_decay=1e-3) # Use AdamW and weight_decay
    logging.info(f"Using pre-trained {config.MODEL_NAME} model. Final layers is trainable.")
    logging.info(f"Optimizer configured with AdamW, LR={args.lr}")

    # Add scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # --- Training Loop ---
    best_val_acc = 0.0
    best_model_path = ""

    # Add these two lines for early stopping
    epochs_no_improve = 0
    early_stop_patience = 7 # Stop after 7 epochs of no improvement
    scaler = amp.GradScaler() # Initialize GradScaler for Mixed Precision

    logging.info("\n--- Starting Training Loop ---")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model_for_training.train() # Use the compiled model
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Autocast operations to float16 where possible
            with amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model_for_training(inputs) # Use the compiled model
                loss = criterion(outputs, labels)
            
            # Scale loss before backward for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_subset)
        epoch_acc = running_corrects.float() / len(train_subset)

        # --- Validation Phase ---
        model_for_training.eval() # Use the compiled model
        val_loss, val_corrects = 0.0, 0
    
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # We can also use autocast in validation for a minor speedup, but it's less critical
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model_for_training(inputs) # Use the compiled model
                    loss = criterion(outputs, labels)
                    
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

                
                    
        val_epoch_loss = val_loss / len(val_subset)
        val_epoch_acc = val_corrects.float() / len(val_subset)

        scheduler.step()
        
        epoch_duration = time.time() - epoch_start_time ### EPOCH DURATION

        ### Added epoch duration to the log line
        logging.info(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        # --- TensorBoard Logging ---
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/validation', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_epoch_acc, epoch)
        # Log the learning rate to see the scheduler work
        writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # -------------------------------------------------

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
            
            # Save the state_dict of the ORIGINAL model, not the compiled one.
            # The original_model's parameters are updated by the compiled_model during training.
            torch.save(original_model.state_dict(), unique_save_path)
            
            shutil.copy(unique_save_path, static_save_path)
            best_model_path = unique_save_path

            # Reset the counter when we find a new best model
            epochs_no_improve = 0
            logging.info(f"  -> 🎉 New best model saved! Val Acc: {best_val_acc:.4f}")            
        else:
            # Increment the counter if no improvement
            epochs_no_improve += 1

        # --- Check for early stopping ---
        if epochs_no_improve >= early_stop_patience:
            logging.info(f"\n--- Early stopping triggered after {early_stop_patience} epochs with no improvement. ---")
            logging.info(f"Best validation accuracy was {best_val_acc:.4f} at epoch {epoch - early_stop_patience + 1}.")
            break # Exit the training loop


    ### FINAL SUMMARY BLOCK
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
    
    writer.close()  # Close the TensorBoard writer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train an image orientation detection model.")
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Directory with upright images.')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_SAVE_DIR, help='Directory to save trained models.')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--workers', type=int, default=config.NUM_WORKERS, help='Number of data loading workers.')
    parser.add_argument('--force-rebuild-cache', action='store_true', help='If set, clears and rebuilds the image cache.')
    
    args = parser.parse_args()
    train(args)