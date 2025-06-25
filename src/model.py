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
        
    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model