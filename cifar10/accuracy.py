import torch

def evaluate(tloader, Model, classes):
    with torch.no_grad():
        correct = 0
        samples = 0
        class_correct = [0 for i in range(10)]
        class_sample = [0 for i in range(10)]
        for images, labels in tloader:
            outputs = Model(images)
#            print(f"outputs.size() is {outputs.size()}")   -> torch([4, 10])
            _, predicted = torch.max(outputs.data, 1)
            # print(f"predicted is {predicted}")    predicted class
            # print(f"labels is {labels}")          actual class

            samples += labels.size(0) # is 4
            correct += (predicted == labels).sum().item() #tensor([true, true, true, false]).sum() -> tensor(3).item() -> 3

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    class_correct[label] += 1
                class_sample[label] += 1

        acc = 100.0 * correct / samples
        print("---------------------------------------")
        print(f'{correct} correct out of {samples}')
        print(f'Accuracy of the network: {acc} %')
        print("---------------------------------------")

        highestacc = []
        for i in range(10):
            if class_sample[i] != 0:
                acc = 100.0 * class_correct[i] / class_sample[i]
                print(f'Accuracy of {classes[i]}: {acc} %')
                highestacc.append(acc)
        print("---------------------------------------")
        print(f"Highest Accuracy at class: {classes[highestacc.index(max(highestacc))]} -> {max(highestacc)} %")
        print("---------------------------------------")
        