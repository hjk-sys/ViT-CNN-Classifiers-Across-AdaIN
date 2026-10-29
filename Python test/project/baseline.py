import torch, torchvision, time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

original_dataset = datasets.CIFAR10(
    root="project/Images",
    train=False,
    download=False,
)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

dataset = ImageFolder(
    root="project/Images/Post_Impression-Batch",
    transform=transform
)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
loader1 = DataLoader(original_dataset, batch_size=32, shuffle=False)

image, label = original_dataset[77]
plt.imshow(image)
plt.show()
# images, labels = dataset[2]
# images = images.permute(1,2,0)
# plt.imshow(images.clamp(0,1))
# plt.show()

# images, labels = next(iter(loader))
# # Checking visual feedback
# image = images[2].cpu()
# image1 = images[2].cpu()
# image = image.permute(1,2,0)
# mean = torch.tensor([0.485,0.456,0.406]).view(1,1,3)
# std = torch.tensor([0.229,0.224,0.225]).view(1,1,3)
# image = image * std + mean
# plt.imshow(image.clamp(0,1))
# # plt.title(original_dataset.classes[label])
# plt.axis("off")
# plt.show()