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
    """ Preprocesses FMNIST data (by rescaling and reshaping it)
   
    Attributes:
        x -- reshaped/rescaled x_train
        y -- targets and labels for x_train, y_Train
    """
    def __init__(self, x, y):
       """ Initilization function

       x  -- fashionMNIST dataset's x_train (values/pictures)
       y  -- fashionMNIST dataset's y_train (targets/labels)
       
       """
       x = x.view(-1, 1, 28, 28)
       x = x.float()/255
       self.x, self.y = x, y
    def __getitem__(self, ix):
       return self.x[ix].to(device), self.y[ix].to(device)
    def __len__(self):
       return len(self.x)

class Models():
    """ This class has all the models needed for the Experiments.
    
        Attributes: 
            kernel_size -- the kernel size of the convolutional layer
            num_filters -- the number of filters the convolutional layer uses
            droupout    -- the dropout percentage
    """
    def __init__(self, kernel_size=3, num_filters=10, dropout=0):
        """ Initialization function for Class. Takes in all possible inputs to make
        running experiments specified by Lab possible

        parameters:
            kernel_size -- the kernel size of the convolutional layer
            num_filters -- the number of filters the convolutional layer uses, automatically 10
            droupout    -- the percentage of nodes dropped after an activation function
        """
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.dropout=dropout

    def make_mlp(self):
        """ Makes an MLP of 3 layers.
        
        return:
            model -- the MLP
        """
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1000), # Layer 1
            nn.ReLU(),
            nn.Linear(1000, 200), # Layer 2
            nn.ReLU(),
            nn.Linear(200, 10), # Layer 3
            ).to(device)
        
        summary(model,  (1, 28, 28))
        return model

    def dense_layer_size(self):
        """ Calculates the input size of the first dense layer after a Convolutional layer based on the
            kernel size and the max_pool operation.

            return:
                The input size of the first dense layer
        """
        conv_size = (28 - self.kernel_size) + 1
        # max_pool layer since it has a kernel size of 2, each side is halved
        # so if the conv layer is (28, 28), maxpool will be (14, 14)
        size_max_pool = conv_size//2
        # To turn everything into 1d, multiply the features and the shape of maxpool to get input size of the first
        # dense layer
        linear_layer_size = (size_max_pool**2)*self.num_filters
        
        return linear_layer_size
        
    def make_cnn(self):
        """ Makes a Convolutional Neural Network with 3 layers. The number of channels and kernel size
            are determined by what is initialized

            return:
                model -- the CNN
        """
        model = nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),   # Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.dense_layer_size(), 200), # Layer 2
            nn.ReLU(),
            nn.Linear(200, 10), # Layer 3
            ).to(device)
        
        summary(model,  (1, 28, 28))
        return model
    
    def make_cnn_dropout(self):
        """ Makes a CNN (which has the same layers as the CNN described above) 
            but with dropouts after every activation function 

            return:
                model -- the dropout CNN
        """
        model = nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),   # Layer 1
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.dense_layer_size(), 200), # Layer 2
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(200, 10), # Layer 3
            ).to(device)
        
        summary(model,  (1, 28, 28))
        return model
    
    def make_cnn_batch_norm(self):
        """ Makes a CNN (which has the same layers as the CNN described above) 
            but with dropouts layers before every activation function

            return:
                model -- the batch normalized CNN
        """
        model = nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size),   # Layer 1
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.dense_layer_size(), 200), # Layer 2
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 10), # Layer 3
            ).to(device)
        
        summary(model,  (1, 28, 28))
        return model

class TrainTestModel():
    """ This class has operations that trains and tests a model. 

        Attributes:
            train_dl       -- the training dataset's dataloader
            test_dl        -- the testing dataset's dataloader
            model          -- the neural network that is being trained and tested
            opt            -- the optimization function used to train the model
            loss_func      -- the loss function used for training the model
            time_to_train  -- the time it took to train the model
            train_accuracy -- the accuracy of the model on the training dataset
            test_accuracy  -- the accuracy of the model on the testing dataset
    """
    def __init__(self, train_dl, test_dl, model, opt, loss_func):
        """ Initialization function
        
        parameters:
            train_dl  -- the training dataset's dataloader
            test_dl   -- the testing dataset's dataloader
            model     -- the neural network that is being trained and tested
            opt       -- the optimization function used to train the model
            loss_func -- the loss function used for training the model
        """
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.time_to_train = 0
        self.train_accuracy = 0
        self.test_accuracy = 0
    
    def train_model(self):
        """ Trains model by training several batches in several epochs """
        n_epochs = 10
        before = time.time()
        for epoch in range(n_epochs):
            print(f"Running epoch {epoch + 1} of {n_epochs}")
            for batch in self.train_dl:
                x, y = batch
                self.train_batch(x, y)
        self.time_to_train = time.time() - before

        return
    
    def train_batch(self, x, y):
        """ Trains model based on batch. Takes loss after and adjusts weights in gradient descent. 

        parameters:
            x -- the set of batch values
            y -- the set of accompanying labels for the batch
        """
        self.model.train()
        # Flush memory
        self.opt.zero_grad()                   
        # Compute loss
        batch_loss = self.loss_func(self.model(x), y)  
        # Compute gradients
        batch_loss.backward()              
        # Make a GD step
        self.opt.step()                         

        return batch_loss.detach().cpu().numpy()
    
    @torch.no_grad() 
    def batch_accuracy(self, x, y,):
        """ Calculates the accuracy of a batch

        parameters:
            x -- the set of batch values
            y -- the set of accompanying labels for the batch
        
        return:
            s -- batch accuracy
        """
        self.model.eval() 
        # Check model prediction
        prediction = self.model(x)
        # Compute the predicted labels for the batch
        argmaxes = prediction.argmax(dim=1) 
        # Compute accuracy
        s = torch.sum((argmaxes == y).float())/len(y)
        return s.cpu().numpy()

    @torch.no_grad()
    def test_model(self):
        """ Tests the model on training and testing data. Stores the results of both accuracies
            in the Object
        """
        test_accuracies = []
        train_accuracies = []

        for batch in self.train_dl:
            x, y = batch
            batch_accuracy = self.batch_accuracy(x, y)
            train_accuracies.append(batch_accuracy)
        self.train_accuracy = np.mean(train_accuracies)

        for batch in self.test_dl:
            x, y = batch
            batch_accuracy = self.batch_accuracy(x, y)
            test_accuracies.append(batch_accuracy)
        
        self.test_accuracy = np.mean(test_accuracies)
    
class Experiments():
    def __init__(self, train_dl, test_dl, loss_func):
        """ Initilization for the experiments class
        
        parameters:
            train_dl  -- the training dataset's dataloader
            test_dl   -- the testing dataset's dataloader
            loss_func -- The loss function used for training, shared amongst all models
        """
        self.train_dl = train_dl
        self.test_dl = test_dl 
        self.loss_func = loss_func
    
    def train_test_model(self, model):
        """ Takes a model, trains and tests it. 
        
        parameters: 
            model -- the neural network model defined by the Models() class
        
        return:
            a tuple of of the time it took to train the model, the model's accuracy on the training data,
            and the model's accuracy on testing data
        """
        opt = Adam(model.parameters(), lr=1e-3)
        trainTestModel = TrainTestModel(self.train_dl, self.test_dl, model, opt, loss_func)
        trainTestModel.train_model()
        trainTestModel.test_model()
        time_taken, train_accuracy, test_accuracy = trainTestModel.time_to_train, \
            trainTestModel.train_accuracy, trainTestModel.test_accuracy

        return time_taken, train_accuracy, test_accuracy
    
    def test_mlp_cnn(self):
        """ Tests an MLP and CNN, printing out their performances """
        mlp = Models().make_mlp()
        cnn = Models().make_cnn()
        summary(cnn, (1, 28, 28))
        mlp_time, mlp_train_accuracy, mlp_test_accuracy = self.train_test_model(mlp)
        cnn_time, cnn_train_accuracy, cnn_test_accuracy = self.train_test_model(cnn)

        print("MLP RESULTS")
        print(f"Time: {mlp_time}, training_acc: {mlp_train_accuracy}, test_acc: {mlp_test_accuracy}")

        print("CNN RESULTS")
        print(f"Time: {cnn_time}, training_acc: {cnn_train_accuracy}, test_acc: {cnn_test_accuracy}")


    def test_kernels(self):
        """ Tests various kernel sizes in CNN and graphs their performance """
        kernel_sizes = [2, 3, 5, 7, 9]
        results = {}
        for size in kernel_sizes: 
            model = Models(kernel_size=size).make_cnn()
            results[size] = self.train_test_model(model)
        print(results)
        self.plot_results(results, "CNN Kernel Size", "Kernel Size Performance")
        
    
    def test_filters(self):
        """ Tests various number of CNN filters (by making a new model everytime) and graphs their performance"""
        num_filters = [5, 10, 15, 20, 25]
        results = {}
        for filters in num_filters:
            model = Models(num_filters=filters).make_cnn()
            results[filters] = self.train_test_model(model)
        print(results)
        self.plot_results(results, "Number of Filters", "# of Filters Performance")

    
    def test_dropouts(self):
        """ Tests various dropout percentages and graphs them """
        dropout_values = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.40, 0.45, 0.50]
        results = {}
        for dropout in dropout_values:
            model = Models(dropout=dropout).make_cnn_dropout()
            results[dropout] = self.train_test_model(model)
        print(results)
        self.plot_results(results, "Dropout Percentage", "Dropout Performance")
    
    def test_bn(self):
        """ Tests a batch normalized CNN and graphs results"""
        model = Models().make_cnn_batch_norm()
        results = self.train_test_model(model)
        print(results)


    def plot_results(self, results, xlabel, title):
        """ Takes results of experiments and produces a graph of training time and accuracy

        parameters:
            results       -- A dictionary with the results of the experiment Keys are the x values in the graph and
                             (values of results are in this format: (time taken to train, training accuracy, test accuracy)). 
            xlabel        -- the label of the x axis in the graph
            title         -- the title of the graph
        """
        # plot time taken to train
        plt.plot(results.keys(), [value[0] for value in results.values()], marker='o', linestyle='solid',
                    linewidth=1.2, markersize=3.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Time (seconds)")
        plt.show()

        # plot test accuracy
        plt.plot(results.keys(), [value[2] for value in results.values()], marker='o', linestyle='solid',
                linewidth=1.2, markersize=3.7, label="test_accuracy")
        # plot train_accuracy
        plt.plot(results.keys(), [value[1] for value in results.values()], marker='o', linestyle='solid',
            linewidth=1.2, markersize=3.7, label="train_accuracy")
        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("Accuracy (%)")
        plt.show()
    
if __name__ == "__main__":
    """ Main function. Creates a Fashion MNIST dataset, reshapes it and runs the Lab's experiments"""
    fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
    fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
    x_train, y_train = fmnist_train.data, fmnist_train.targets
    x_test, y_test = fmnist_test.data, fmnist_test.targets

    train_dataset = FMNISTDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = FMNISTDataset(x_test, y_test)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    loss_func = nn.CrossEntropyLoss()
    experiments = Experiments(train_dl, test_dl, loss_func)
    experiments.test_mlp_cnn()
    experiments.test_kernels()
    experiments.test_filters()
    experiments.test_bn()
    experiments.test_dropouts()
