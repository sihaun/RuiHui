import torch.nn as nn
from torchvision.models import efficientnet_b7

def tf_efficientnet_b7(num_classes):
    """
    transfer learning efficientnet_b7 model
    """
    net = efficientnet_b7(pretrained=True)
    
    # Turn off gradient calculation for all layers
    for param in net.parameters():
        param.requires_grad = False
    
    # Modify the final classifier
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    
    return net

