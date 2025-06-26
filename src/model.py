# /src/model.py

import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES

def get_orientation_model(pretrained=True):
    """
    Loads a pre-trained ResNet18 model and replaces its final
    layer for our 4-class orientation task.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Use fine-tuning
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer.
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512), # Add an intermediate layer
        nn.ReLU(),
        nn.Dropout(p=0.5), # Add a dropout layer with 50% probability
        nn.Linear(512, NUM_CLASSES) # The final output layer
    )

    return model