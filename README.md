# ViT-CNN-Classifiers-Across-AdaIN

Compares how a CNN (fine-tuned **ResNet-50**) and a Vision Transformer (**ViT-Base-16**) hold up when the same images are shown in different artistic styles.

The test set is the 10,000 CIFAR-10 test images. Each image is also rendered in six painting styles using AdaIN style transfer, giving 7 versions of the test set (original + 6 styles) and 70,000 images in total. Both models are evaluated on every version.

For each model and version, `Evaluate.py` reports overall accuracy, per-class accuracy, a confusion matrix, per-class precision / recall / F1, parameter count, and inference speed (images/sec).

## Setup

Developed with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Build the dataset

The images are not stored in this repo. One script downloads CIFAR-10 and generates every stylized version:

```bash
python "Python test/project/generate_dataset.py"
```

This runs 60,000 style transfers, so it takes a long time. It skips images that already exist, so you can stop it and resume later.

The AdaIN checkpoint and the six reference paintings are included in `Python test/project/Style_Transfer/`.

| Output folder | Reference painting |
|---|---|
| `Base Copy` | original CIFAR-10 images (no style) |
| `abstract-Batch` | `abstract.jpeg` |
| `cubism-Batch` | `cubism.jpg` |
| `oil_painting-Batch` | `oil_painting.jpeg` |
| `Post_Impression-Batch` | `starry_night.jpg` |
| `sketch-Batch` | `sketch.jpeg` |
| `watercolor-Batch` | `watercolor.jpg` |

Each folder contains one subfolder per CIFAR-10 class, under `Python test/project/Images/`.

## 2. Trained models

This project evaluates two existing CIFAR-10 checkpoints from Hugging Face. They are about 420 MB combined and are not included in this repo. Download these two files and place them in `Python test/project/models/`:

| File | Model | Source |
|---|---|---|
| `finetune_resnet_cifar_model.pt` | ResNet-50 fine-tuned on CIFAR-10 | [AnjanSB/finetune-resnet-cifar10](https://huggingface.co/AnjanSB/finetune-resnet-cifar10/tree/main) |
| `vit_base_16_cifar10_original.pth` | ViT-Base-16 fine-tuned on CIFAR-10 | [Yurim0507/vit-base-16-cifar10-unlearning](https://huggingface.co/Yurim0507/vit-base-16-cifar10-unlearning) |

The ViT repository contains 11 checkpoints. Only the `original` one (trained on all 10 classes) is used here.

## 3. Run the experiments

The scripts use paths relative to the `Python test` folder, so run them from there:

```bash
cd "Python test"
python project/ResnetInput.py   # evaluate ResNet-50
python project/ViTInput.py      # evaluate ViT-Base-16
```

Both scripts evaluate one version of the test set per run, and it is chosen inside the script. Each script has one `ImageFolder(...)` block per dataset. The oil painting block is active by default. To evaluate a different version, comment out the active block and uncomment the one you want.

To evaluate the original images, point the `root` at `project/Images/Base Copy` instead:

```python
dataset = ImageFolder(
    root="project/Images/Base Copy",
    transform=transform
)
```

Available dataset folders: `Base Copy`, `abstract-Batch`, `cubism-Batch`, `oil_painting-Batch`, `Post_Impression-Batch`, `sketch-Batch`, `watercolor-Batch`.

The scripts use an Apple Silicon GPU (MPS) when one is available and fall back to the CPU otherwise.

## Repository layout

```
Python test/project/
├── generate_dataset.py   # builds the 70,000-image dataset
├── Evaluate.py           # accuracy, confusion matrix, precision/recall/F1, speed
├── Resnet.py             # ResNet-50 fine-tuning model
├── ResnetInput.py        # runs ResNet-50 on the datasets
├── ViTModel.py           # ViT-Base-16 model loading
├── ViTInput.py           # runs ViT-Base-16 on the datasets
├── baseline.py           # dataset loading / sanity checks
├── Style_Transfer/       # AdaIN code, checkpoint, and style paintings
├── figures/              # figures
├── models/               # trained weights (not in the repo)
└── Images/               # generated dataset (not in the repo)
```

## Credits

- ResNet-50 CIFAR-10 checkpoint: [AnjanSB/finetune-resnet-cifar10](https://huggingface.co/AnjanSB/finetune-resnet-cifar10) (MIT license)
- ViT-Base-16 CIFAR-10 checkpoint: [Yurim0507/vit-base-16-cifar10-unlearning](https://huggingface.co/Yurim0507/vit-base-16-cifar10-unlearning) (MIT license)
- Style transfer: AdaIN (Huang and Belongie, 2017)
- Dataset: CIFAR-10 (Krizhevsky, 2009)
