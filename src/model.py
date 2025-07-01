# /src/model.py

import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES

def get_orientation_model(model_name, pretrained=True):
    """
    Loads a pre-trained EfficientNet model and replaces its final
    layer for our 4-class orientation task.
    """
    if "efficientnet_v2_s" in model_name:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
    elif "efficientnet_v2_m" in model_name:
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_m(weights=weights)
    elif "efficientnet_v2_l" in model_name:
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_l(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers for fine-tuning
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.features[-2].parameters():
        param.requires_grad = True
        
    # Get the number of input features for the classifier
    num_ftrs = model.classifier[1].in_features
    
    # Replace the final fully connected layer.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, NUM_CLASSES),
    )

    return model