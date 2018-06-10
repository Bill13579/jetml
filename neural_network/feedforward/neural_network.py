import jetm as jm
from jetml.neural_network.feedforward.layer import InputLayer, HiddenLayer, OutputLayer

class NeuralNetwork:
    ACTIVATION_FUNCTIONS = {
        "relu": {
            "_": jm.relu,
            "derivative": jm.relu_p
        },
        "leaky-relu": {
            "_": jm.leaky_relu,
            "derivative": jm.leaky_relu_p
        },
        "sigmoid": {
            "_": jm.sigmoid,
            "derivative": jm.sigmoid_p
        }
    }

    def __init__(self, layers, activation_functions=("relu", "sigmoid")):
        if len(layers) < 2:
            raise self.InitializationException("A feedforward neural network has to have at least 2 layers")
        self.layers = layers
        self.__verify_activation_function(activation_functions)
        self.activation_functions = activation_functions
    
    def __verify_activation_function(self, activation_functions):
        for i in range(len(activation_functions)):
            af = activation_functions[i]
            if af not in NeuralNetwork.ACTIVATION_FUNCTIONS.keys() and af != "identity":
                raise Exception("Unknown activation function \"" + af + "\"")

    def __activation_function(self, layer_index, prime=False):
        aflength = len(self.activation_functions)
        activation_function = None
        if aflength == len(self.layers):
            activation_function = self.activation_functions[layer_index]
        elif aflength == 2:
            activation_function = self.activation_functions[1] if layer_index == len(self.layers)-1 else self.activation_functions[0]
        elif aflength == 1:
            activation_function = self.activation_functions[0]
        func = None
        if activation_function == "identity":
            func = jm.identity
        else:
            if not prime:
                func = NeuralNetwork.ACTIVATION_FUNCTIONS[activation_function]["_"]
            else:
                func = NeuralNetwork.ACTIVATION_FUNCTIONS[activation_function]["derivative"]
        return func

    def __apply_activation_function(self, vector, layer_index, prime=False):
        func = self.__activation_function(layer_index, prime)
        vector = vector.copy()
        for row in range(vector.shape[0]):
            vector[row, 0] = func(vector[row, 0])
        return vector
    
    def remove_layer(self, index):
        if index < 1 or index > len(self.layers)-2:
            raise self.RemovalFailedException("The layer specified is an input layer or an output layer and cannot be removed")
        del self.layers[index]
    
    def add_hidden_layer(self, new):
        self.layers.insert(-1, new)
        
    def swap_input_layer(self, new):
        self.layers[0] = new
    
    def swap_output_layer(self, new):
        self.layers[-1] = new
    
    def feedforward(self, data):
        activations = data.copy()
        total_layers = len(self.layers)
        for i in range(1, total_layers):
            layer = self.layers[i]
            weights = layer.weights
            biases = layer.biases
            z = weights * activations + biases
            activations = self.__apply_activation_function(z, i)
        return activations

    class RemovalFailedException(Exception):
        pass
    
    class InitializationException(Exception):
        pass

class LGNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, activation_function=("relu", "sigmoid")):
        if len(layers) < 2:
            raise self.InitializationException("A feedforward neural network has to have at least 2 layers")
        generated_layers = []
        generated_layers.append(InputLayer(layers[0]))
        if len(layers) > 2:
            for l in range(1, len(layers)-1):
                generated_layers.append(HiddenLayer(layers[l], layers[l-1]))
        generated_layers.append(OutputLayer(layers[-1], layers[-2]))
        super().__init__(generated_layers, activation_function)

