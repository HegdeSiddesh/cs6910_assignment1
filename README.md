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

The default values of the parameters for the optimizers are:
1. SGD(learning_rate = 0.001, weight_decay = 0.0)
2. Momentum(learning_rate = 0.001, gamma = 0.001, weight_decay = 0.0)
3. NAG(learning_rate = 0.001, gamma = 0.9)
4. RMSPROP(learning_rate = 0.001, gamma = 0.001, epsilon = 1e-8, weight_decay = 0.0)
5. Adam(learning_rate = 0.001, epsilon = 1e-8, weight_decay = 0.0, beta1 = 0.9, beta2 = 0.999)
6. NAdam(learning_rate = 0.001, epsilon = 1e-8, weight_decay = 0.0, beta1 = 0.9, beta2 = 0.999)

Select the optimizer of choice and create an object for the same. The parameters for the selected optimizer can be set using the "updateParameters(dictionary)" function of the optimizer classes, where the input is a dictionary containing key value pairs correcponding to the parameters to be set. Each optimizer class consists the function which can compute gradient descent using the optimizer.

### Creating object for activation function to be used:

The following choices of activation functions are available:
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

The class FeedForwardNeuralNetwork has the logic for the forward and backpropogation algorithm. An object of this class must be created, which would require the following parameters as input:
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

An example for the above is as follows:
```python
ffnn_model = FeedForwardNeuralNetwork(layers, optimizer, loss, activation, output_activation, epoch_count, batch_size, initialization = "Xavier-Normal", train_losses_list = train_losses, train_accuracy_list = train_accuracy, val_losses_list = val_losses, val_accuracy_list = val_accuracy)
```


### Fitting data using object of FeedForwardNeuralNetwork class:

Once the object for the FeedForwardNeuralNetwork class is created, the "fit" function needs to be called which will fit the required data using the model. The data needed to be passed to the fit function is (x_train, y_train, x_val, y_val) which is the training data and training label and validation data and validation labels. The labels for Fashion-MNIST are the OneHotEncoded label values for each datapoint.

After the model runs for the mentioned number of epochs, the "fit" function returns 4 parameters which are (training loss, training accuracy, validation loss, validation accuracy).

An example for the above is as follows:
```python
ffnn_model.fit(x_train, y_train, x_val, y_val)
```

### Making new predictions:

When we have data points which need to classified, we can use the "predict" function of the FeedForwardNeuralNetwork object which has been trained on the train data. The input to the "predict" function will be a list of datapoints, and the returned value will be predicted probabilities for each datapoint given by the trained model.

An example for the above is as follows:
```python
y_test_predictions = ffnn_model.predict(x_test)
```

### Testing prediction accuracy:

Using the "accuracy" function, the accuracy of the model can be evaluated. The parameters required for the "accuracy" function are the true and predicted labels.

An example for the above is as follows:
```python
#Here y_test_labels and y_test_predictions are the true and predicted labels
#If labels are in OneHotEncoded form or predicted probabilities, use numpy.argmax(y) to get the class true/predicted labels first
test_accuracy = accuracy(y_test_labels, y_test_predictions)
```

### Creating new Optimizers

The process for creating new optimizers is to create a class for the new optimizer which would have the following functions in it:
1. __init__(input_parameters) : This function takes the input parameters passed by the user and assigns them to variables associated to the object for the class. Set default parameters here (if any) for the optimizer.
2. initialize() : This function is to initialize any weights, biases or intermediate parameters which would be required for repeated use in every epoch. Example: Momentum optimizer required update_history_w which is the update history for the weights of the network. Such variables can be initialized before the run of the network.
3. optimizer_name(): Returns a string with the name of the optimizer
4. set_weight_decay(weight_decay) : Sets the weight decay(if any) for the optimizer
5. set_learning_rate(learning_rate) : Sets the learning rate(if any) for the optimizer
6. set_initial_parameters(dictionary) : Input is a dictionary with the key value pairs of the parameters to be initialized
7. update_parameters(weights, biases, dw, db, layer_count): This function performs the gradient descent step using the weights and biases and their computed derivatives. Also the number of layers present in the network has to be passed.


```python
class OptimizerName():
  def __init__(self, learning_rate = 0.001, weight_decay = 0.0, parameter_1 = None, parameter_2 = None):
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.parameter_1 = parameter_1
    self.parameter_2 = parameter_2

  def initialize(self, all_layers):
    #Initialize the required parameters here

  def set_learning_rate(self, learning_rate):
    #Set learning rate here if necessary 
    self.learning_rate = learning_rate
  
  def set_weight_decay(self, weight_decay):
    #Set weight decay here if necessary 
    self.weight_decay = weight_decay
  
  def optimizer_name(self):
    return "name"

  def set_initial_parameters(self, parameters):
      self.parameter_1 = parameters["parameter_1"]
      self.parameter_2 = parameters["parameter_2"]

  def update_parameters(self, weights, biases, dw, db, layers):

    for i in range(len(layers)+1):  
      #Enter gradient descent logic here
      
    return weights, biases
```


