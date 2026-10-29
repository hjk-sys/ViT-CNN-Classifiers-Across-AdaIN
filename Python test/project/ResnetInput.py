import torch, time
from Resnet import ResNet50FineTune
from Evaluate import evaluate_all
from torchvision import transforms

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

# Evaluates every dataset listed in Evaluate.py that isn't saved yet
evaluate_all(model, device, "resnet", transform)
