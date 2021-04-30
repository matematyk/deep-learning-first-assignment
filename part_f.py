# We will store the weights in a D x c matrix, where D is the number of features, and c is the number of classes
#weights = (...) # TODO: Fill in, be sure to have the right shape!
import numpy as np
from loader import MyDataset

def softmax(z):
    return (np.exp(z).T /np.sum(np.exp(z),axis=1)).T


def predict(weights, X):
    return softmax(np.dot(X, weights))

from numpy import linalg as LA
def compute_loss_and_gradients(weights, X, y, l2_reg):
    loss = -np.trace(np.dot(np.log(predict(weights, X)), y.T)) / y.shape[0] + l2_reg * np.sum(weights**2)

    return loss
    
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


def fit(l2_reg=0.001, n_epochs=30, lr=1e-4, batch_size=100):
  transformed_dataset = MyDataset(txt_path='data/labels.csv', img_dir='data/',
                  transform=transforms.Compose([
                      transforms.Resize(28),
                      transforms.CenterCrop(28),
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ]))
  
  dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=0)
  losses = []
  weights = np.zeros([784, 6])
  for i in range(n_epochs):
    for i_batch, sample_batched in enumerate(dataloader):
      X = sample_batched['image'].reshape(-1, 28 * 28) / 255
      loss = compute_loss_and_gradients(weights, X , sample_batched['label'], l2_reg)
      losses.append(loss)
      
  return weights, losses

weights, losses = fit(l2_reg=0.001, n_epochs=30, lr=1e-4, batch_size=100)
