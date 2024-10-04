import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import time

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


class TrainTestModel():
    def __init__(self, train_dl, test_dl, model, opt, loss_func):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.time_to_train = 0
        self.accuracy = 0
    
    def train_model(self):
        losses, accuracies, n_epochs = [], [], 2
        before = time.time()
        for epoch in range(n_epochs):
            print(f"Running epoch {epoch + 1} of {n_epochs}")

            epoch_losses, epoch_accuracies = [], []
            for batch in self.train_dl:
                x, y = batch
                batch_loss = self.train_batch(x, y)
                epoch_losses.append(batch_loss)
            epoch_loss = np.mean(epoch_losses)

            for batch in self.train_dl:
                x, y = batch
                batch_acc = self.batch_accuracy(x, y)
                epoch_accuracies.append(batch_acc)
            epoch_accuracy = np.mean(epoch_accuracies)
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)

        self.time_to_train = time.time() - before
        # these are for internal use only. Used to check how well the model is improving.
        print(f"Losses: {losses}")
        print(f"Accuracies: {accuracies}")

        return
    
    def train_batch(self, x, y):
        self.model.train()
        self.opt.zero_grad()                    # Flush memory
        batch_loss = self.loss_func(self.model(x), y)  # Compute loss
        batch_loss.backward()              # Compute gradients
        self.opt.step()                         # Make a GD step
        return batch_loss.detach().cpu().numpy()
    
    @torch.no_grad() # This decorator is used to tell PyTorch that nothing here is used for training
    def batch_accuracy(self, x, y,):
        self.model.eval() 
        prediction = self.model(x) # Check model prediction
        argmaxes = prediction.argmax(dim=1) # Compute the predicted labels for the batch
        s = torch.sum((argmaxes == y).float())/len(y) # Compute accuracy
        return s.cpu().numpy()

    @torch.no_grad()
    def test_model(self):
        accuracies = []
        for batch in self.test_dl:
            x, y = batch
            batch_accuracy = self.batch_accuracy(x, y)
            accuracies.append(batch_accuracy)
        
        self.accuracy = np.mean(accuracies)
        return
    
class Models():
    def __init__(self, kernel_size=3, num_filters=10, dropout=0.25):
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.dropout=0.25

    def make_mlp(self):
       return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1000), # Layer 1
            nn.ReLU(),
            nn.Linear(1000, 10) # Layer 2
            ).to(device)
    
    def make_cnn(self):
        conv_size = (28 - self.kernel_size) + 1
        size_max_pool = conv_size//2
        linear_layer_size = (size_max_pool**2)*self.num_filters

        print(linear_layer_size)
        # linear_layer_size = ((conv_size//2)**2)*num_filters
        # print(linear_layer_size)
        return nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),   # Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(linear_layer_size, 10),   # Layer 2
            ).to(device)
    
    def make_cnn_dropout(self):
      # where should dropout be?
      return nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),   # Layer 1
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(10816, 10),   # Layer 2
            ).to(device)
        
    
    # def make_cnn_batch_norm(self):
    #     nn.Sequential()
    #     return

class Experiments():
    def __init__(self, train_dl, test_dl, loss_func):
        self.train_dl = train_dl
        self.test_dl = test_dl 
        self.loss_func = loss_func
    
    def model_performance(self, model):
        opt = Adam(model.parameters(), lr=1e-3)
        trainTestModel = TrainTestModel(self.train_dl, self.test_dl, model, opt, loss_func)
        trainTestModel.train_model()
        trainTestModel.test_model()
        time_taken, accuracy = trainTestModel.time_to_train, trainTestModel.accuracy

        return time_taken, accuracy
    
    def test_mlp_cnn(self):
        mlp = Models().make_mlp()
        cnn = Models().make_cnn()
        summary(cnn, (1, 28, 28))
        # mlp_time, mlp_accuracy = self.model_performance(mlp)
        cnn_time, cnn_accuracy = self.model_performance(cnn)

        # print("MLP RESULTS")
        # print(mlp_time, mlp_accuracy)
        print("CNN RESULTS")
        print(cnn_time, cnn_accuracy)


    def test_kernels(self):
        kernel_sizes = [2, 3, 5, 7, 9 ]
        results = {}
        for size in kernel_sizes: 
            model = Models(kernel_size=size).make_cnn()
            results[size] = self.model_performance(model)
            # trainTestModel = TrainTestModel(train_dl, model, opt, loss_func)
            # trainTestModel.train_model()
            # trainTestModel.test_model()
            # time_taken, accuracy = trainTestModel.time_to_train, trainTestModel.accuracy
            # results[size] = (accuracy, time_taken)
        print(results)
        return results
    
    def test_filters(self):
        num_filters = [5, 10, 15, 20, 25]
        results = {}
        for filters in num_filters:
            model = Models(num_filters=filters).make_cnn()
            results[filters] = self.model_performance(model)
        print(results)
        return results
    
    def test_dropouts(self):
        dropout_values = [0.02, 0.03, 0.1, 0.25, 0.5, 0.75]
        results = {}
        for dropout in dropout_values:
            model = Models(dropout=dropout).make_cnn_dropout()
            results[dropout] = self.model_performance(model)
        print(results)
        return
    def test_bn(self):
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
    
    loss_func = nn.CrossEntropyLoss()
    # study #1
    experiments = Experiments(train_dl, test_dl, loss_func)
    # experiments.test_mlp_cnn()
    experiments.test_kernels()
    experiments.test_dropouts()
    # experiments.test_bn()

    # #study #1
    # mlp = Models().make_mlp()
    # # summary(mlp, (1, 28, 28))
    # opt = Adam(mlp.parameters(), lr=1e-3)
    
    # print("mlp")
    # # trained_mlp = TrainModel(train_dl, mlp, opt, loss_func).train_model()
    # cnn = Models().make_cnn()
    # # summary(cnn, (1, 28, 28))
    # opt = Adam(cnn.parameters(), lr=1e-3)
    # print("cnn")
    # trained_cnn = TrainModel(train_dl, cnn, opt, loss_func).train_model()

    # # test kernel size
    # for i in kernel_size:
