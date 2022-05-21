from cgi import test
from re import X
from black import out
from numpy import size
from sklearn.model_selection import learning_curve
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pylab as plt
import torch.nn as nn
from torchvision.transforms import ToTensor

train_datasets = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())
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

model = NeuralNet().to(device)

learning_rate = 0.001

loss_fn = nn.CrossEntropyLoss()
torch.set_grad_enabled(True) 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(train_dataloader, model, loss_fn, optimizer):
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.step()
        optimizer.zero_grad()

        if batch%100==0:
            loss = loss.item()
            print(f'loss:  {loss}, batch: {batch}')

def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

        pred = model(x)

        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1)==y).type(torch.floadt).sum().item()
    
    test_loss/=num_batches
    correct/=size
    print(f'Test Error: \n Accuracy: {(100*correct): 0.1f}%, Avg loss: {test_loss:>8f} \n')

epochs = 10
for t in range(epochs):
    print(f'Epoch {t+1}\n ---------------------------')
    train(train_dataloader, model)
    test(test_dataloader, model, loss_fn)

print("Done!")