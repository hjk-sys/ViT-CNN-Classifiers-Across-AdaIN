import torch, torchvision, time
import matplotlib.pyplot as plt
from Resnet import ResNet50FineTune
from Evaluate import evaluation
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

pre_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Transforming to fit pretrained Imagnet architecture
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
original_dataset = datasets.CIFAR10(
    root="project/Images",
    train=False,
    download=False,
)
image, label = original_dataset[102]

""" Starry Night Dataset"""
# dataset = ImageFolder(
#     root="project/Images/Post_Impression-Batch",
#     transform=transform
# )
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


start_time = time.perf_counter()
model = ResNet50FineTune(num_classes=10)

model.to(device)
checkpoint = torch.load("project/models/finetune_resnet_cifar_model.pt", map_location=device)

model.load_state_dict(checkpoint)

model.eval()

print("ResNet-50 Model loaded successfully!")
end_time = time.perf_counter()
print(f"Model Loading Time: {round((end_time - start_time),2)} seconds")

evaluation(model,device,loader)