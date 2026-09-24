import torch, time

def evaluation(model, device, loader):
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

    return images, labels
