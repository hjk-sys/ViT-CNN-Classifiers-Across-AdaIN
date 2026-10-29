import torch, time
from ViTModel import load_model
from Evaluate import evaluate_all
from torchvision import transforms

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

# Apple sillicon device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

start_time = time.perf_counter()
model = load_model("project/models/vit_base_16_cifar10_original.pth", num_classes=10, device=device)
end_time = time.perf_counter()
print("ViT_base_16 Model Successfully Loaded!")
print(f"Model Loading Time: {round((end_time - start_time),2)} seconds")

# Evaluates every dataset listed in Evaluate.py that isn't saved yet
evaluate_all(model, device, "vit", transform)
