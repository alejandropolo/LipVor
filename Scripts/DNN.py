import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init

class DNN(nn.Module):
    """
    Deep Neural Network (DNN) class.

    This class constructs a deep neural network with specified layers and activation functions.

    Attributes:
        depth (int): The number of layers in the network.
        layers (nn.Sequential): The sequential container of layers and activation functions.

    Methods:
        __init__(layers, activations):
            Initializes the DNN with the given layers and activations.
        
        forward(x):
            Defines the forward pass of the network.
    """

    def __init__(self, layers, activations):
        """
        Initializes the DNN with the given layers and activations.

        Args:
            layers (list of int): A list containing the number of neurons in each layer.
            activations (list of str): A list containing the activation functions for each layer.
                Supported activation functions are 'relu', 'sigmoid', 'tanh', and 'leakyrelu'.

        Raises:
            Exception: If the number of activations is greater than the number of layers minus one.
            Exception: If an invalid activation function is specified.
        """
        super(DNN, self).__init__()
        
        # Check length of layers and activations
        if len(layers) - 1 < len(activations):
            raise Exception('The number of activations must be equal or less than the number of layers')

        # Parameters
        self.depth = len(layers) - 1
        
        # Deploy layers with activations
        layer_list = []
        for i in range(self.depth):
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            # Determine the activation function for this layer
            if i < len(activations):
                activation_function = activations[i].lower()
            elif i == self.depth - 1:
                activation_function = None
            else:
                activation_function = 'tanh'

            # Create the activation layer based on the specified activation function
            if activation_function == 'relu':
                activation = nn.ReLU()
            elif activation_function == 'sigmoid':
                activation = nn.Sigmoid()
            elif activation_function == 'tanh':
                activation = nn.Tanh()
            elif activation_function == 'leakyrelu':
                activation = nn.LeakyReLU()
            elif activation_function is None:
                activation = None
            else:
                raise Exception(f'Invalid activation function: {activation_function}')
            
            if activation is not None:
                layer_list.append(('activation_%d' % i, activation))
        
        layer_dict = OrderedDict(layer_list)
        
        # Deploy layers
        self.layers = nn.Sequential(layer_dict)
        
        # He Initialization for linear layers
        for name, module in self.layers.named_children():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='sigmoid')
        
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        out = self.layers(x)
        return out