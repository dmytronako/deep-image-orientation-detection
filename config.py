import torch

# --- Caching ---
# If True, the script will pre-process and cache all image rotations.
# This is much faster for training, but requires disk space.
# If False, rotations are done on-the-fly in memory.
USE_CACHE = True
CACHE_DIR = "data/cache"


# --- Dataloader and Preprocessing ---
DATA_DIR = "data/upright_images"
IMAGE_SIZE = 224
BATCH_SIZE = 512
# Set num_workers to 0 if you are on Windows, otherwise > 0
NUM_WORKERS = 16 # Or 12. Experiment with this value.

# --- Model Configuration ---
MODEL_SAVE_DIR = "models"
MODEL_NAME = "orientation_model_v8" # Incremented version
NUM_CLASSES = 4  # 0°, 90°, 180°, 270°

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 25

# --- Prediction Settings ---
# A dictionary to map class indices to human-readable actions
CLASS_MAP = {
    0: "Image is correctly oriented (0°).",
    1: "Image needs to be rotated 90° Counter-Clockwise to be correct.",
    2: "Image needs to be rotated 180° to be correct.",
    3: "Image needs to be rotated 90° Clockwise to be correct."
}