import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.optim import adam
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
x_train, y_train = fmnist_train.data, fmnist_train.targets
x_test, y_test = fmnist_test.data, fmnist_test.targets

# what train goes here?
train_dl = DataLoader(x_train, batch_size=32, shuffle=True)

class Study():
    def __init__():
        return
    def train_model(self, train_dl, model):
        opt = adam(model.parameters(), lr=1e-3)
        loss_func = nn.CrossEntropyLoss()
        losses, accuracies, n_epochs = [], [], 5
        for epoch in range(n_epochs):
            print(f"Running epoch {epoch + 1} of {n_epochs}")
            epoch_losses, epoch_accuracies = [], []
        for batch in train_dl:
            x, y = batch
            batch_loss = self.train_batch(x, y, model, opt, loss_func)
            epoch_losses.append(batch_loss)
            epoch_loss = np.mean(epoch_losses)
        for batch in train_dl:
            x, y = batch
            batch_acc = self.accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
            epoch_accuracy = np.mean(epoch_accuracies)
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
        
        return losses, accuracies
    
    def train_batch(self, x, y, model, opt, loss_func):
        model.train() # We’ll see the meaning of this line later today
        opt.zero_grad() # Flush memory
        batch_loss = loss_func(model(x), y) # Compute loss
        batch_loss.backward() # Compute gradients
        opt.step() # Make a GD step
        return batch_loss.detach().cpu().numpy() # Removes grad, sends data to cpu, converts tensor to array
    
    @torch.no_grad() # This decorator is used to tell PyTorch that nothing here is used for training
    def accuracy(self, x, y, model):
        model.eval() # We’ll see the meaning of this line later today
        prediction = model(x) # Check model prediction
        argmaxes = prediction.argmax(dim=1) # Compute the predicted labels for the batch
        s = torch.sum((argmaxes == y).float())/len(y) # Compute accuracy
        return s.cpu().numpy()


class Model():
    def __init__(self, num_units=0, kernel_size=0, num_filters=0):
        self.num_units = num_units
        self.kernel_size = kernel_size
        self.num_filters = num_filters

    def make_mlp(self):
        nn.Sequential(nn.Linear(),
            nn.ReLU(),
            nn.Linear()
            )
        return

    def make_cnn(self):
        nn.Sequential(nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(-1, self.num_filters, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(3200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
            )
        return
    
    def make_cnn_dropout(self):
        nn.sequential()
        return
    
    def make_cnn_batch_norm(self):
        nn.sequential()
        return

    
# FMNIST Dataset
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.view(-1,28*28).float()/255
        self.x, self.y = x, y
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    def __len__(self):
        return len(self.x)






def graph():
    return

if __name__ == "__main__":
    # do fifferent tudies
    kernel_size = [2, 3, 5, 7, 9 ]
    num_filters = [5, 10, 15, 20, 25]
