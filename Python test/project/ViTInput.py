import torch.nn as nn
import torch, torchvision, time
from ViTModel import load_model
from Evaluate import evaluation
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# Transforming to fit pretrained Imagnet architecture similar to resnet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# Dataset
""" Post Impressionism Dataset """
# dataset = ImageFolder(
#      root="project/Images/Post_Impression-Batch",
#      transform=transform
#  )
""" Sketch Dataset """
# dataset = ImageFolder(
#      root="project/Images/sketch-Batch",
#      transform=transform
#  )
""" Watercolor Dataset """
# dataset = ImageFolder(
#     root="project/Images/watercolor-Batch",
#     transform=transform
# )
""" Cubism Dataset """
# dataset = ImageFolder(
#     root="project/Images/cubism-Batch",
#     transform=transform
# )
""" Abstract Dataset """
# dataset = ImageFolder(
#     root="project/Images/abstract-Batch",
#     transform=transform
# )
""" Oil Painting Dataset """
dataset = ImageFolder(
    root="project/Images/oil_painting-Batch",
    transform=transform
)
# Batch configuration
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Apple sillicon device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Example usage:
start_time = time.perf_counter()
model = load_model("project/models/vit_base_16_cifar10_original.pth", num_classes=10, device=device)
end_time = time.perf_counter()
print("ViT_base_16 Model Successfully Loaded!")
print(f"Model Loading Time: {round((end_time - start_time),2)} seconds")

evaluation(model,device,loader)

