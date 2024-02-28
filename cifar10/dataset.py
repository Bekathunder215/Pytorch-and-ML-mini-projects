import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# CustomTransform=transforms.ToTensor()

CustomTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def getDatasets():
    train_dataset = torchvision.datasets.CIFAR10(root="../data",
                                                train=True,
                                                transform=CustomTransform,
                                                download=False)
    test_dataset = torchvision.datasets.CIFAR10(root="../data",
                                                train=False,
                                                transform=CustomTransform,
                                                download=False)
    return train_dataset, test_dataset

def getLoader(batchSize):
    train, test = getDatasets()
    train_loader = DataLoader(train, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test, batch_size=batchSize, shuffle=False)
    # for plotting, but is unfinished
    # img, label = next(iter(test_loader))
    # print(img[0].dtype)
    # fig = plt.figure(figsize=(4, 4))
    # fig.add_subplot(2, 2, 1)
    # plt.imshow(transforms.ToPILImage()(img[0]))
    # plt.axis("off")
    # plt.title(classes[label[0].item()])
    # fig.add_subplot(2, 2, 2)
    # plt.imshow(transforms.ToPILImage()(img[1]))
    # plt.axis("off")
    # plt.title(classes[label[1].item()])
    # fig.add_subplot(2, 2, 3)
    # plt.imshow(transforms.ToPILImage()(img[2]))
    # plt.axis("off")
    # plt.title(classes[label[2].item()])
    # fig.add_subplot(2, 2, 4)
    # plt.imshow(transforms.ToPILImage()(img[3]))
    # plt.axis("off")
    # plt.title(classes[label[3].item()])
    # plt.show()
    return train_loader, test_loader

