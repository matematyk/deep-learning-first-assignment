import torch.nn as nn
from loader import MyDataset
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import log_loss

MB_SIZE = 512//2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        modules = []

        modules.append(nn.Conv2d(4, 16, 3, padding=1))
        modules.append(nn.BatchNorm2d(16))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Conv2d(16, 8, 3, padding=1))
        modules.append(nn.BatchNorm2d(8))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Flatten())
        modules.append(nn.Linear(8 * 7 * 7, 60))
        modules.append(nn.ReLU())
        modules.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)

        return x
class ExerciseTrainer(object):
    def __init__(self):
        transformed_dataset = MyDataset(txt_path='data/labels.csv', img_dir='data/',
                transform=transforms.Compose([
                    transforms.Resize(28),
                    transforms.CenterCrop(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
        transformed_validation = MyDataset(txt_path='data/validation.csv', img_dir='data/',
                transform=transforms.Compose([
                    transforms.Resize(28),
                    transforms.CenterCrop(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
        
        self.trainloader = torch.utils.data.DataLoader(
            transformed_dataset, batch_size=MB_SIZE, shuffle=True, num_workers=4)

       
        self.testloader = torch.utils.data.DataLoader(
            transformed_validation, batch_size=1, shuffle=False, num_workers=4)

    def additional_metrics(self, y_hat, y):
        for i in range(0,5):
            for j in range(0,9)
                y[i][j]

    def own_loss(y, y_hat):
            loss = 0
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    loss += y[i,j] * torch.log(y_hat[i,j]) + (1- y[i,j])*torch.log(1-y_hat[i,j])
            return -loss / y.shape[0]


    def train(self):
        net = Net()



        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        


        for epoch in range(2):
            running_loss = 0.0
            for inputs, labels in self.trainloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = own_loss(labels, outputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            correct = 0
            total = 0
            net.eval()
            acc_add = 0
            with torch.no_grad():
                for inputs, labels in self.testloader:
                    images, labels = inputs, labels
                    outputs = net(images)
                    total += labels.size(0)
                    metric = self.additional_metrics(outputs, labels)
                    acc_add += metric
                    

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, acc_add/ (2*total)))

def main():
    trainer = ExerciseTrainer()
    trainer.train()


if __name__ == '__main__':
    main()

