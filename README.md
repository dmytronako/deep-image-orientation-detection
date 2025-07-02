# Image Orientation Detector

This project implements a deep learning model to detect the orientation of images and determine the rotation needed to correct them. It uses a pre-trained EfficientNetV2 model from PyTorch, fine-tuned for the task of classifying images into four orientation categories: 0°, 90°, 180°, and 270°.

The model achieves **97.53% accuracy** on the validation set.

## Training Performance and Model History

This model was trained on a single NVIDIA RTX 4080 GPU, taking approximately **3 hours and 20 minutes** to complete.

The final model is using `EfficientNetV2-S`, but the project evolved through several iterations:

- **ResNet18:** Achieved ~90% accuracy with a model size of around 30MB.
- **ResNet50:** Improved accuracy to 95.26% with a model size of ~100MB.
- **EfficientNetV2-S:** Reached the "final" (for now) accuracy of **97.53%** with ~80MB.

## How It Works

The model is trained on a dataset of images, where each image is rotated by 0°, 90°, 180°, and 270°. The model learns to predict which rotation has been applied. The prediction can then be used to determine the correction needed to bring the image to its upright orientation.

The four classes correspond to the following rotations:

- **Class 0:** Image is correctly oriented (0°).
- **Class 1:** Image needs to be rotated 90° Counter-Clockwise to be correct.
- **Class 2:** Image needs to be rotated 180° to be correct.
- **Class 3:** Image needs to be rotated 90° Clockwise to be correct.

## Dataset

The model was trained on several datasets:

- **Microsoft COCO Dataset:** A large-scale object detection, segmentation, and captioning dataset ([link](https://cocodataset.org/)).
- **AI-Generated vs. Real Images:** A dataset from Kaggle ([link](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)) was included to make the model aware of the typical orientations on different compositions found in art and illustrations.
- **Personal Images:** A small, curated collection of personal photographs to include unique examples and edge cases.

The combined dataset consists of **45,726** unique images. Each image is augmented by being rotated in four ways (0°, 90°, 180°, 270°), creating a total of **182,904** samples. This augmented dataset was then split into **146,323 samples for training** and **36,581 samples for validation**.

## Project Structure

```
image_orientation_detector/
├───.gitignore
├───config.py                 # Main configuration file for paths, model, and hyperparameters
├───convert_to_onnx.py        # Script to convert the PyTorch model to ONNX format
├───predict.py                # Script for running inference on new images
├───README.md                 # This file
├───requirements.txt          # Python dependencies
├───train.py                  # Main script for training the model
├───data/
│   ├───upright_images/       # Directory for correctly oriented images
│   └───cache/                # Directory for cached, pre-rotated images (auto-generated)
├───models/
│   └───best_model.pth        # The best trained model weights
└───src/
    ├───caching.py            # Logic for creating the image cache
    ├───dataset.py            # PyTorch Dataset classes
    ├───model.py              # Model definition (EfficientNetV2)
    └───utils.py              # Utility functions (e.g., device setup, transforms)
```

## Usage

### Getting Started

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Prediction

To predict the orientation of an image or a directory of images, there's a `predict.py` script.

- **Predict a single image:**

  ```bash
  python predict.py --input_path /path/to/image.jpg
  ```
- **Predict all images in a directory:**

  ```bash
  python predict.py --input_path /path/to/directory/
  ```

The script will output the predicted orientation for each image.

### ONNX Export and Prediction

This project also includes exporting the trained PyTorch model to the ONNX (Open Neural Network Exchange) format. This allows for faster inference, especially on hardware that doesn't have PyTorch installed.

To convert a `.pth` model to `.onnx`, provide the path to the model file:

```bash
python convert_to_onnx.py path/to/model.pth
```

This will create a `model.onnx` file in the same directory.

To predict image orientation using the ONNX model:

- **Predict a single image:**

  ```bash
  python predict_onnx.py --input_path /path/to/image.jpg
  ```
- **Predict all images in a directory:**

  ```bash
  python predict_onnx.py --input_path /path/to/directory/
  ```

#### ONNX GPU Acceleration (Optional)

For even better performance on NVIDIA GPUs, you can install the GPU-enabled version of ONNX Runtime.

```bash
pip install onnxruntime-gpu
```

Make sure you have a compatible CUDA toolkit installed on your system. The `predict_onnx.py` script will automatically try to use the CUDA provider if it's available.

#### Performance Comparison (PyTorch vs. ONNX)

For a dataset of 5055 images, the performance on a RTX 4080 running in **single-thread** was:

- **PyTorch (`predict.py`):** 135.71 seconds
- **ONNX (`predict_onnx.py`):** 60.83 seconds

This demonstrates a significant performance gain of approximately **55.2%** when using the ONNX model for inference.

### Training

This model learns to identify image orientation by training on a dataset of images that you provide. For the model to learn effectively, provide images that are correctly oriented.

**Place Images in the `data/upright_images` directory**: All images must be placed in the `data/upright_images` directory. The training script will automatically generate rotated versions (90°, 180°, 270°) of these images and cache them for efficient training.

The directory structure should look like this:

```
data/
└───upright_images/
    ├───image1.jpg
    ├───image2.png
    └───...
```

### Configure the Training

All training parameters are centralized in the `config.py` file. Before starting the training, review and adjust the settings to match the hardware and dataset.

Key configuration options in `config.py`:

- **Paths and Caching**:

  - `TRAIN_IMAGES_PATH`: Path to upright images. Defaults to `data/upright_images`.
  - `CACHE_PATH`: Directory where rotated images will be cached. Defaults to `data/cache`.
  - `USE_CACHE`: Set to `True` to use the cache on subsequent runs, significantly speeding up data loading but takes a lot of disk space.
- **Model and Training Hyperparameters**:

  - `MODEL_NAME`: The name of the model architecture to use (e.g., `EfficientNetV2S`).
  - `IMAGE_SIZE`: The resolution to which images will be resized (e.g., `224` for 224x224 pixels).
  - `BATCH_SIZE`: Number of images to process in each batch. Adjust based on GPU's VRAM.
  - `NUM_EPOCHS`: The total number of times the model will iterate over the entire dataset.
  - `LEARNING_RATE`: The initial learning rate for the optimizer.

### Start Training

Once all data is in place and the configuration is set,  start training the model by running the `train.py` script:

```bash
python train.py
```

- **First Run**: The first time the script runs, it will preprocess and cache the dataset. This may take a while depending on the size of the dataset.
- **Subsequent Runs**: Later runs will be much faster as they will use the cached data.
- **Monitoring**: Use TensorBoard to monitor training progress by running `tensorboard --logdir=runs`.

### Monitoring with TensorBoard

The training script is integrated with TensorBoard to help visualize metrics and understand the model's performance. During training, logs are saved in the `runs/` directory.

To launch TensorBoard, run the command:

```bash
tensorboard --logdir=runs
```

This will start a web server, open the provided URL (usually `http://localhost:6006`) in the browser to view the dashboard.

In TensorBoard, you can track:

- **Accuracy:** `Accuracy/train` and `Accuracy/validation`
- **Loss:** `Loss/train` and `Loss/validation`
- **Learning Rate:** `Hyperparameters/learning_rate` to see how it changes over epochs.
