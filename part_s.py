import torch.nn as nn
from loader import MyDataset
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

MB_SIZE = 512//2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        modules = []

        modules.append(nn.Conv2d(3, 32, 3, padding=1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Conv2d(32, 8, 3, padding=1))
        modules.append(nn.BatchNorm2d(8))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Flatten())
        modules.append(nn.Linear(8 * 7 * 7, 6))
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
                    transforms.GaussianBlur(),
                    transforms.CenterCrop(28),
                    transforms.ToTensor()
                ]), binary=False)
        transformed_validation = MyDataset(txt_path='data/validation.csv', img_dir='data/',
                transform=transforms.Compose([
                    transforms.Resize(28),
                    transforms.GaussianBlur(),
                    transforms.CenterCrop(28),
                    transforms.ToTensor()
                ]), binary=False )
        
        self.trainloader = torch.utils.data.DataLoader(
            transformed_dataset, batch_size=MB_SIZE, shuffle=True, num_workers=4)

       
        self.testloader = torch.utils.data.DataLoader(
            transformed_validation, batch_size=1, shuffle=False, num_workers=4)

    def additional_metrics(self, y_hat, y):
        metric = torch.zeros(y_hat.shape[0])
        for i in range(y_hat.shape[0]):
            y_new = torch.zeros(6)
            y_new[torch.argsort(-y_hat[i])[0:2]]=1
            met = 0 
            for j in range(y_hat.shape[1]):
                if y_new[j] == y[i][j] and y_new[j] == 1:
                    met += 1
            metric[i] = met
        return metric

    def own_loss(self, y, y_hat):
            loss = 0
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    loss += y[i,j] * torch.log(y_hat[i,j]) + (1- y[i,j])*torch.log(1-y_hat[i,j])
            return -loss / y.shape[0]


    def train(self):
        net = Net()

        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        for epoch in range(20):
            running_loss = 0.0
            for inputs, labels in self.trainloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.own_loss(labels, outputs)
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
                total, acc_add / (2*total)))

def main():
    trainer = ExerciseTrainer()
    trainer.train()


if __name__ == '__main__':
    main()

