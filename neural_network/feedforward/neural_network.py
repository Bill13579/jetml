import jetm as jm
import jetml.math as Math
from jetml.neural_network.feedforward.layer import InputLayer, HiddenLayer, OutputLayer

class NeuralNetwork:
    ACTIVATION_FUNCTIONS = {
        "relu": {
            "_": Math.relu,
            "derivative": Math.relu_p
        },
        "leaky-relu": {
            "_": Math.leaky_relu,
            "derivative": Math.leaky_relu_p
        },
        "sigmoid": {
            "_": Math.sigmoid,
            "derivative": Math.sigmoid_p
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
            func = Math.identity
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
    
    def __feedforward(self, data):
        activations_record = {}
        z_record = {}
        activations = data.copy()
        activations_record[0] = activations
        total_layers = len(self.layers)
        for i in range(1, total_layers):
            layer = self.layers[i]
            weights = layer.weights
            biases = layer.biases
            z = weights * activations + biases
            activations = self.__apply_activation_function(z, i)
            z_record[i] = z
            activations_record[i] = activations
        return activations, z_record, activations_record

    def backpropagate(self, training_data, learning_rate=0.1):
        total_layers = len(self.layers)
        data = training_data.dataset
        for i in range(len(data)):
            pred, z_record, activations_record = self.__feedforward(data[i][0])
            expected = data[i][1]
            diff = pred - expected
            dcost_dpred = 2 * diff
            diff * 2
            dcost_dla = dcost_dpred
            for j in range(total_layers-1, 0, -1):
                z = z_record[j]
                c_layer = self.layers[j]
                # Calculate some partial derivatives
                da_dz = self.__apply_activation_function(z, j, True)
                dz_dla = jm.matrix.transpose(c_layer.weights)
                dz_dw = activations_record[j-1]
                # Adjust weights
                dcost_dw = Math.multiply_across(Math.multiply_combinations(da_dz, dz_dw), dcost_dla)
                c_layer.weights = c_layer.weights - dcost_dw * learning_rate
                # Adjust biases
                dcost_db = jm.matrix.entrywise(da_dz, dcost_dla)
                c_layer.biases = c_layer.biases - dcost_db * learning_rate
                # Calculate the derivative of the cost function with respect to the activations of the last layer
                dcost_dla = dz_dla * jm.matrix.entrywise(da_dz, dcost_dla)

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

