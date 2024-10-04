import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FMNISTDataset(Dataset):
   def __init__(self, x, y):
       x = x.view(-1, 1, 28, 28)
       x = x.float()/255
       self.x, self.y = x, y
   def __getitem__(self, ix):
       return self.x[ix].to(device), self.y[ix].to(device)
   def __len__(self):
       return len(self.x)

class TrainModel():
    def __init__(self, train_dl, model, opt, loss_func):
        self.train_dl = train_dl
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
    
    def train_model(self):
        losses, accuracies, n_epochs = [], [], 5
        for epoch in range(n_epochs):
            print(f"Running epoch {epoch + 1} of {n_epochs}")

            epoch_losses, epoch_accuracies = [], []
            for batch in self.train_dl:
                x, y = batch
                batch_loss = self.train_batch(x, y, self.model, self.opt, self.loss_func)
                epoch_losses.append(batch_loss)
            epoch_loss = np.mean(epoch_losses)

            for batch in self.train_dl:
                x, y = batch
                batch_acc = self.accuracy(x, y, self.model)
                epoch_accuracies.append(batch_acc)
            epoch_accuracy = np.mean(epoch_accuracies)

            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
        print(f"Losses: {losses}")
        print(f"Accuracies: {accuracies}")
        return losses, accuracies
    
    def train_batch(self, x, y, model, opt, loss_func):
        model.train()
        opt.zero_grad()                    # Flush memory
        batch_loss = loss_func(model(x), y)  # Compute loss
        batch_loss.backward()              # Compute gradients
        opt.step()                         # Make a GD step

        return batch_loss.detach().cpu().numpy()
    
    @torch.no_grad() # This decorator is used to tell PyTorch that nothing here is used for training
    def accuracy(self, x, y, model):
        model.eval() # We’ll see the meaning of this line later today
        prediction = model(x) # Check model prediction
        argmaxes = prediction.argmax(dim=1) # Compute the predicted labels for the batch
        s = torch.sum((argmaxes == y).float())/len(y) # Compute accuracy
        return s.cpu().numpy()


class Model():
    def __init__(self, num_units=2, kernel_size=3, num_filters=64):
        self.num_units = num_units
        self.kernel_size = kernel_size
        self.num_filters = num_filters

    def make_mlp(self):
       return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1000), # Layer 1
            nn.ReLU(),
            nn.Linear(1000, 10) # Layer 2
            ).to(device)
    
    def make_cnn(self):
        return nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),   # Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(10816, 10),   # Layer 2
            ).to(device)
    
    def make_cnn_dropout(self):
        nn.Sequential()
        return
    
    def make_cnn_batch_norm(self):
        nn.Sequential()
        return


if __name__ == "__main__":
    kernel_size = [2, 3, 5, 7, 9 ]
    num_filters = [5, 10, 15, 20, 25]

    fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
    fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
    x_train, y_train = fmnist_train.data, fmnist_train.targets
    x_test, y_test = fmnist_test.data, fmnist_test.targets

    train_dataset = FMNISTDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = FMNISTDataset(x_test, y_test)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    #study #1
    mlp = Model().make_mlp()
    # summary(mlp, (1, 28, 28))
    opt = Adam(mlp.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    print("mlp")
    trained_mlp = TrainModel(train_dl, mlp, opt, loss_func).train_model()
    cnn = Model().make_cnn()
    # summary(cnn, (1, 28, 28))
    opt = Adam(cnn.parameters(), lr=1e-3)
    print("cnn")
    trained_cnn = TrainModel(train_dl, cnn, opt, loss_func).train_model()
