import os
import sys
from torchvision.datasets import CIFAR10

# all paths are built from this file's location, so it works from any folder
here = os.path.dirname(os.path.abspath(__file__))   
sys.path.insert(0, os.path.join(here, "Style_Transfer"))  

from Style_Transfer.infer import run_infer

images = os.path.join(here, "Images")
style_dir = os.path.join(here, "Style_Transfer", "styles")
ckpt = os.path.join(here, "Style_Transfer", "weights", "cktp-style-net-30000.tar")

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


styles = {
    "abstract-Batch": "abstract.jpeg",
    "cubism-Batch": "cubism.jpg",
    "oil_painting-Batch": "oil_painting.jpeg",
    "Post_Impression-Batch": "starry_night.jpg",
    "sketch-Batch": "sketch.jpeg",
    "watercolor-Batch": "watercolor.jpg",
}

for folder in ["Base Copy"] + list(styles):
    for c in classes:
        os.makedirs(os.path.join(images, folder, c), exist_ok=True)

dataset = CIFAR10(root=images, train=False, download=True)

total = len(dataset)   # for a quick test, temporarily set this to 20
for i in range(total):
    image, label = dataset[i]
    class_name = dataset.classes[label]

    base_path = os.path.join(images, "Base Copy", class_name, f"{i}.png")
    if not os.path.exists(base_path):
        image.save(base_path)

    for folder, painting in styles.items():
        save_path = os.path.join(images, folder, class_name, f"{i}.png")
        if os.path.exists(save_path):
            continue   
        styled = run_infer(image, os.path.join(style_dir, painting), ckpt_dir=ckpt)
        styled.save(save_path)

    if i % 10 == 0:
        print(f"{i}/{total} images done ({round(i / total * 100, 2)}%)")