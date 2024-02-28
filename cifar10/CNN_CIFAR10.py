import torch
import torch.nn as nn
import dataset
from CNN_MODEL import CNN
from train import training
from accuracy import evaluate

#hyper parameters
num_epochs = 30
batch_size = 4
learning_rate = 0.0005
PATH="./cifar10/trainedmodel.pt"
# if i want to load, train, or evaluate i'll make var true or 1 
load = 1
train_toggle = 0
evaluate_toggle = 1

#dataset
train, test = dataset.getLoader(batchSize=batch_size)
classes = dataset.classes
img, labels = next(iter(test)) # img.shape -> 4, 3, 32, 32

#CNN
model = CNN()

if(load):
    model.load_state_dict(torch.load(PATH))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_total_steps = len(train)

if(train_toggle):
    model = training(epochs=num_epochs,
                 total_steps=num_total_steps,
                 loader=train,
                 Optimizer=optimizer,
                 Criterion=criterion,
                 Model=model)

if(train_toggle):
    torch.save(model.state_dict(), PATH)

if(evaluate_toggle):
    evaluate(test, model, classes=classes)