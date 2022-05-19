from cgi import test
from re import X
from black import out
from sklearn.model_selection import learning_curve
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pylab as plt
import torch.nn as nn
from torchvision.transforms import ToTensor

train_datasets = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

# image, label = train_datasets[0]
# plt.imshow(image, cmap='gray')
# plt.show()
# print('label: ', label)

batch_size = 100

train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
print(f'Device: {device}')

input_size = 28*28
output_size = 10

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__() # .
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.flatten(out)
        return out

learning_rate = 0.001
