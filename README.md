# CS6910 Assignment 1 : Implementation of FeedForward Neural Network

The objective of this assignment is to implement a feedforward neural network architecture along with backpropogation and with the choice of selecting from a set of available optimizers, activation functions and loss functions mentioned below. The model if to be tested on Fashion-MNIST dataset. The assignment is divided into multiple sections and the report for the performance of the model on the datset is reported in the wandb report [here](https://wandb.ai/hegdesiddesh/Assignment_1/reports/Assignment-1--VmlldzoxNTQwMjA5).


## Training the model

### Creating object for optimizer to be used:

The following choices of optimizers are available:
- SGD
- Mometum
- NAG
- RMSPROP
- Adam
- NAdam

Select the optimizer of choice and create an object for the same. The parameters for the selected optimizer can be set using the "updateParameters(dictionary)" function of the optimizer classes, where the input is a dictionary containing key value pairs correcponding to the parameters to be set. Each optimizer class consists the function which can compute gradient descent using the optimizer.

### Creating object for activation function to be used:

The following choices of optimizers are available:
- Sigmoid
- Tanh
- Relu
- Softmax

Select the activation function of choice and create an object for the same. Each class consists the functions corresponding to calculate the activation and its derivate given an input.

### Creating object for loss function to be used:

The following choices of optimizers are available:
- Cross entropy loss
- Squared loss

Select the loss function of choice and create an object for the same. Each class consists the functions corresponding to calculate the loss and its gradient given an input.


### Creating an object for FeedForwardNeuralNetwork class:

The class FeedForwardNeuralNetwork consists of the logic for the forward and backpropogation algorithm. An object of this class must be created, which would require the following parameters as input:
1. List with the hidden layer sizes (list size indicates the number of hidden layers for the network)
2. Object for the optimizer to be used.
3. Object for the loss function to be used.
4. Object for the activation function to be used at the hidden layers.
5. Object for the activation function to be used at the output layer.

Below are the optional parameters which can be passed:

7. Number of epochs (default 1000)
8. Batch size (default 1024)
9. Weight initialization (default "Random")
10. Log onto wand (default 0)
11. Log onto console (default 1)
12. List for saving training loss per epoch (default None)
13. List for saving training accuracy per epoch (default None)
14. List for saving validation loss per epoch (default None)
15. List for saving validation accuracy per epoch (default None)


### Fitting data using object of FeedForwardNeuralNetwork class:

Once the object for the FeedForwardNeuralNetwork class is created, the "fit" function needs to be called which will fit the required data using the model. The data needed to be passed to the fit function is (x_train, y_train, x_val, y_val) which is the training data and training label and validation data and validation labels.

After the model runs for the mentioned number of epochs, the "fit" function returns 4 parameters which are (training loss, training accuracy, validation loss, validation accuracy).



