import torch, time, os, csv
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder

RESULTS_DIR = "project/results"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Every dataset the models are evaluated on (None = the original CIFAR-10 test set)
DATASETS = {
    "baseline": None,
    "alpha0": "project/Images/alpha0-Batch",              # control: same AdaIN pipeline, alpha = 0 (no style)
    "resize_only": "project/Images/resize_only-Batch",    # control: 32 -> 256 -> 32 resize only, no AdaIN
    "oil_painting": "project/Images/oil_painting-Batch",
    "abstract": "project/Images/abstract-Batch",
    "post_impression": "project/Images/Post_Impression-Batch",
    "watercolor": "project/Images/watercolor-Batch",
    "sketch": "project/Images/sketch-Batch",
    "cubism": "project/Images/cubism-Batch",
    # Extra reference paintings, to check results aren't a quirk of one painting
    "cubism_b": "project/Images/cubism_b-Batch",
    "cubism_c": "project/Images/cubism_c-Batch",
    "sketch_b": "project/Images/sketch_b-Batch",
    "sketch_c": "project/Images/sketch_c-Batch",
}


def evaluate_all(model, device, model_name, transform):
    """Runs evaluation() on every dataset that is fully generated and not saved yet.
    To redo one, delete its file in project/results/."""
    for name, folder in DATASETS.items():
        save_as = f"{RESULTS_DIR}/{model_name}__{name}"
        if os.path.exists(save_as + ".npz"):
            print(f"skip {name}: already saved")
            continue
        if folder is None:
            dataset = datasets.CIFAR10(root="project/Images", train=False, download=False, transform=transform)
        elif not os.path.isdir(folder):
            print(f"skip {name}: {folder} not generated yet")
            continue
        else:
            dataset = ImageFolder(root=folder, transform=transform)
            if len(dataset) != 10000:
                # Generation still running or interrupted: never save a partial result
                print(f"skip {name}: only {len(dataset)}/10000 images")
                continue

        print(f"\n----- {model_name} on {name} -----")
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        evaluation(model, device, loader, save_as)


def save_results(save_as, dataset, labels, predictions, confusion_matrix, seconds, parameters):
    """Saves everything results.py needs, so no number has to be copied from the terminal."""
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    labels, predictions = np.array(labels), np.array(predictions)
    if isinstance(dataset, ImageFolder):
        # Stylized images are saved as <CIFAR-10 test index>.png
        image_ids = np.array([int(os.path.basename(path)[:-4]) for path, _ in dataset.samples])
    else:
        image_ids = np.arange(len(dataset))

    np.savez(save_as + ".npz", predictions=predictions, labels=labels,
             confusion=confusion_matrix.numpy().astype(np.int64), seconds=seconds, parameters=parameters)
    with open(save_as + "_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true", "predicted"])
        for image, true, predicted in sorted(zip(image_ids, labels, predictions)):
            writer.writerow([image, CLASSES[true], CLASSES[predicted]])
    print(f"Saved to {save_as}.npz and {save_as}_predictions.csv")


def evaluation(model, device, loader, save_as=None):
    images, labels = next(iter(loader))


    corrects = 0
    totals = 0
    class_correct = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0
    }
    class_total = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0
    }


    all_predictions = []
    all_labels = []
    confusion_matrix = torch.zeros(10,10)

    start_time1 = time.perf_counter()              
    for images, labels in loader:


        images = images.to(device)
        labels = labels.to(device)


        with torch.no_grad():
            outputs = model(images)


        _, predictions = torch.max(outputs, 1)


        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


        for prediction, label in zip(predictions, labels):
            confusion_matrix[label][prediction] += 1
            class_total[label.item()] += 1
            if prediction == label:
                class_correct[label.item()] += 1




        correct = (predictions == labels).sum().item()


        corrects += correct
        totals += labels.size(0)


    end_time1 = time.perf_counter()




    accuracy = corrects / totals
    print(f"On the testing data, the model acchieved {accuracy * 100}% accuracy!")

    classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck"
    ]

    for i in range(len(classes)):
        accuracy = class_correct[i]/class_total[i] * 100
        print(f"{classes[i]}:, {round(accuracy,2)}%")


    print(confusion_matrix)


    for i in range(len(classes)):


        TP = confusion_matrix[i][i]


        FN = confusion_matrix[i].sum() - TP


        FP = confusion_matrix[:, i].sum() - TP


        TN = confusion_matrix.sum() - TP - FP - FN


        precision = (TP/(TP+FP))
        recall = (TP/(TP+FN))
        F1_score = (2*((precision*recall)/(precision+recall)))


        print(f"{classes[i]}: P = {round(precision.item()*100,2)}%, R = {round(recall.item()*100,2)}%, F1 = {round(F1_score.item()*100,2)}%")




    images_per_second = totals / (end_time1 - start_time1)
    parameters = sum(p.numel() for p in model.parameters())


    print(f"Parameters: {round(parameters,2)}")
    print(f"Images/sec: {round(images_per_second,2)}")
    print(f"Evaluation time: {round((end_time1 - start_time1),2)} seconds")

    if save_as is not None:
        save_results(save_as, loader.dataset, all_labels, all_predictions, confusion_matrix,
                     end_time1 - start_time1, parameters)

    return images, labels
