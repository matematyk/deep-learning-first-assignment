import torch.nn as nn
from loader import MyDataset
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder

MB_SIZE = 2048

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.shape[0],) + self.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        modules = []

        modules.append(nn.Conv2d(3, 16, 3, padding=1))
        modules.append(nn.BatchNorm2d(16))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Conv2d(16, 8, 3, padding=1))
        modules.append(nn.BatchNorm2d(8))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Flatten())
        modules.append(nn.Linear(8 * 7 * 7, 135)) #135 class
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

    def mapping(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        labels = []
        for i in y.shape[0]:
            str1 = " "
            labels.append((str1.join(y[i]))
        enc.fit(labels)

        return enc        

    def own_loss(self, y_acc, y_hat):
        J = 0 
        for k in range(y_hat.shape[0]): 
            for i in range(y_hat.shape[1]): #6
                for j in range(y_hat.shape[2]): #10
                    J  += y_hat[k][i][j]*(j - y_acc[k][i])** 2

        return J

    def additional_metrics(self, y_acc, y_hat):
        metric = torch.zeros(y_hat.shape[0])
        soft = torch.argmax(y_hat, dim=2)
        for i in range(y_acc.shape[0]):
            ids = torch.argsort(-y_acc[i])[0:2]
            if torch.eq(soft[i][ids], y_acc[i][ids]).sum() == 2:
                metric[i] = 1
            else:
                metric[i] = 0
        return metric



    def train(self):
        net = Net()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)


        for epoch in range(2):
            running_loss = 0.0
            for inputs, labels in self.trainloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.own_loss(labels, outputs)
                metric = self.additional_metrics(labels, outputs)
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
                    metric = self.additional_metrics(labels, outputs)
                    acc_add += metric
                    

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, acc_add/ total))

def main():
    trainer = ExerciseTrainer()
    trainer.train()


if __name__ == '__main__':
    main()

