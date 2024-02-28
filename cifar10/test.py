import torch
from CNN_MODEL import CNN
from PIL import Image
from torchvision import transforms
from CNN_CIFAR10 import classes

PATH="./cifar10/trainedmodel.pt"
img_PATH="./cifar10/test_imgs/isthisafrog.jpeg"
transform = transforms.Compose([transforms.PILToTensor(),
                               transforms.ConvertImageDtype(float),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                  ])



#CNN
model = CNN()
model.load_state_dict(torch.load(PATH))

with Image.open(img_PATH) as im:
    img = im.resize((32, 32)).convert("RGB")
    tensorimg = transform(img)
    tensorimg = tensorimg.to(torch.float32)
    outputs = model(tensorimg)
    _, predicted = torch.max(outputs.data, 1)
    print(f"predicted is {predicted}")
    print(f"predicted class is {classes[predicted.item()]}")