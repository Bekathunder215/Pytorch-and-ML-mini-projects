import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import requests
from PIL import Image
from io import BytesIO

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#imgsize 4, 3, 32, 32

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)  #out -> 30*30*6
        self.pool = nn.MaxPool2d(2,2)  #out -> 15*15*6
        self.conv2 = nn.Conv2d(6,12,5,2) #out -> 6*6*12  after pool 3*3*12
        self.fc1 = nn.Linear(3*3*12, 224)
        self.fc2 = nn.Linear(224, 112)
        self.fc3 = nn.Linear(112, 56)
        self.fc4 = nn.Linear(56, 10)
        
    def forward(self, x):
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)

        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 3*3*12)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

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
        

from PIL import Image
from torchvision import transforms

PATH="./cifar10/api/trainedmodel.pt"
img_PATH1="./cifar10/test_imgs/isthisafrog.jpeg"
transform = transforms.Compose([transforms.PILToTensor(),
                               transforms.ConvertImageDtype(float),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                  ])

#CNN
model = CNN()
model.load_state_dict(torch.load(PATH))

def predict(Model=model, img_PATH="https://www.nationalgeographic.com/content/dam/expeditions/transports/islander-ii/new-day-2-islander-ii-jan23-1000x666.jpg"):
    image=requests.get(img_PATH)
    with Image.open(BytesIO(image.content)) as im:
        img = im.resize((32, 32)).convert("RGB")
        tensorimg = transform(img)
        tensorimg = tensorimg.to(torch.float32)
        # print(type(tensorimg))
        # tensorimg = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(tensorimg)
        # print(f"tensorimg is {tensorimg}")
        # print(f"tensorimg.shape is {tensorimg.shape}")
        outputs = Model(tensorimg)
        _, predicted = torch.max(outputs.data, 1)
        # print(f"predicted is {predicted}")
        # print(f"predicted class is {classes[predicted.item()]}")
        # print(outputs[0,:].tolist())
        json_result = {"Prediction_Str":outputs[0,:].tolist()[predicted.item()],
                       "Class": classes[predicted.item()],
                       "Other_classes": outputs[0,:].tolist()}
        # print(json_result)
        return json.dumps(json_result)

if __name__ == "__main__":
    result = predict()
    print(result)