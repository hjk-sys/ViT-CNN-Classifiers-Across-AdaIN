import torch.nn as nn
import torch
from torchvision.models import ViT_B_16_Weights, vit_b_16

def vit_base_16(num_classes, **kwargs):
    
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: str, num_classes: int = 10, device: str = 'cpu'):
    model = vit_base_16(num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model