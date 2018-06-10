import jetm as jm
import jetml.math as Math
from jetml.neural_network.feedforward import NeuralNetwork

class Backpropagation:
    def __init__(self, neural_network, learning_rate=0.1):
        self.nn = neural_network
        self.learning_rate = learning_rate
    
    def __activation_function(self, layer_index, prime=False):
        aflength = len(self.nn.activation_functions)
        activation_function = None
        if aflength == len(self.nn.layers):
            activation_function = self.nn.activation_functions[layer_index]
        elif aflength == 2:
            activation_function = self.nn.activation_functions[1] if layer_index == len(self.nn.layers)-1 else self.nn.activation_functions[0]
        elif aflength == 1:
            activation_function = self.nn.activation_functions[0]
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

    def __feedforward(self, data):
        activations_record = {}
        z_record = {}
        activations = data.copy()
        activations_record[0] = activations
        total_layers = len(self.nn.layers)
        for i in range(1, total_layers):
            layer = self.nn.layers[i]
            weights = layer.weights
            biases = layer.biases
            z = weights * activations + biases
            activations = self.__apply_activation_function(z, i)
            z_record[i] = z
            activations_record[i] = activations
        return activations, z_record, activations_record

    def train(self, training_data):
        total_layers = len(self.nn.layers)
        data = training_data.dataset
        for i in range(len(data)):
            pred, z_record, activations_record = self.__feedforward(data[i][0])
            expected = data[i][1]
            diff = pred - expected
            dcost_dpred = 2 * diff
            dcost_dla = dcost_dpred
            for j in range(total_layers-1, 0, -1):
                z = z_record[j]
                c_layer = self.nn.layers[j]
                # Calculate some partial derivatives
                da_dz = self.__apply_activation_function(z, j, True)
                dz_dla = jm.matrix.transpose(c_layer.weights)
                dz_dw = activations_record[j-1]
                # Adjust weights
                dcost_dw = Math.multiply_across(Math.multiply_combinations(da_dz, dz_dw), dcost_dla)
                c_layer.weights = c_layer.weights - dcost_dw * self.learning_rate
                # Adjust biases
                dcost_db = jm.matrix.entrywise(da_dz, dcost_dla)
                c_layer.biases = c_layer.biases - dcost_db * self.learning_rate
                # Calculate the derivative of the cost function with respect to the activations of the last layer
                dcost_dla = dz_dla * jm.matrix.entrywise(da_dz, dcost_dla)