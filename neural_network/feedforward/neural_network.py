import jetmath as jm
import jetml.vars as Vars
from jetml.neural_network.feedforward.layer import InputLayer, HiddenLayer, OutputLayer

class NeuralNetwork:
    def __init__(self, layers, activation_functions=("relu", "sigmoid")):
        if len(layers) < 2:
            raise self.InitializationException("A feedforward neural network has to have at least 2 layers")
        self.layers = layers
        self.__verify_activation_function(activation_functions)
        self.activation_functions = activation_functions
    
    def __verify_activation_function(self, activation_functions):
        for i in range(len(activation_functions)):
            af = activation_functions[i]
            if af not in Vars.ACTIVATION_FUNCTIONS.keys() and af != "identity":
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
                func = Vars.ACTIVATION_FUNCTIONS[activation_function]["_"]
            else:
                func = Vars.ACTIVATION_FUNCTIONS[activation_function]["derivative"]
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
    
    def copy(self):
        return NeuralNetwork([l.copy() for l in self.layers], self.activation_functions)
    
    def save(self, path):
        af = ",".join(self.activation_functions)
        nn_str = ""
        nn_str += "af=" + af
        nn_str += " d" + str(self.layers[0].dimensionality)
        for layer_i in range(1, len(self.layers)):
            l = self.layers[layer_i]
            layer_str = " "
            dimensionality = l.dimensionality
            layer_str += "d" + str(dimensionality) + "w"
            weights = l.weights.tolist()
            weights_mat_rows = []
            for r in weights:
                weights_mat_rows.append("n".join(str(w) for w in r))
            layer_str += "N".join(weights_mat_rows)
            layer_str += "b"
            biases = jm.matrix.matrix.transpose(l.biases).tolist()
            layer_str += "N".join(str(b) for b in biases[0])
            nn_str += layer_str
        f = open(path, "wb")
        f.write(nn_str.encode('ascii'))
        f.close()

    @staticmethod
    def from_file(path):
        try:
            metadata = {}
            layers = []
            nn_file = open(path, "rb")
            b = nn_file.read(1).decode('ascii')
            tmp = ""
            c_meta = True
            c_layer = False
            tmp_layer_dimensionality = None
            tmp_layer_weights = None
            tmp_layer_biases = None
            while True:
                if b == " " or b == "":
                    if c_meta:
                        combo = tmp.split("=")
                        metadata[combo[0]] = combo[1]
                        c_meta = False
                    elif c_layer and tmp_layer_weights is None:
                        layers.append(InputLayer(int(tmp)))
                        c_layer = False
                    elif c_layer:
                        tmp_layer_biases[-1].append(float(tmp))
                        layers.append(HiddenLayer(tmp_layer_dimensionality, layers[-1].dimensionality, jm.matrix.matrix(tmp_layer_weights), jm.matrix.matrix(tmp_layer_biases)))
                        tmp_layer_dimensionality = None
                        tmp_layer_weights = None
                        tmp_layer_biases = None
                        c_layer = False
                    tmp = ""
                elif b == "=":
                    c_meta = True
                    tmp += b
                elif not c_meta:
                    if b == "d":
                        c_layer = True
                    elif b == "w":
                        tmp_layer_dimensionality = int(tmp)
                        tmp_layer_weights = [[]]
                        tmp = ""
                    elif b == "b":
                        tmp_layer_weights[-1].append(float(tmp))
                        tmp_layer_biases = [[]]
                        tmp = ""
                    elif b == "n":
                        if tmp_layer_biases is None:
                            tmp_layer_weights[-1].append(float(tmp))
                        else:
                            tmp_layer_biases[-1].append(float(tmp))
                        tmp = ""
                    elif b == "N":
                        if tmp_layer_biases is None:
                            tmp_layer_weights[-1].append(float(tmp))
                            tmp_layer_weights.append([])
                        else:
                            tmp_layer_biases[-1].append(float(tmp))
                            tmp_layer_biases.append([])
                        tmp = ""
                    else:
                        tmp += b
                else:
                    tmp += b
                if b == "":
                    break
                b = nn_file.read(1).decode('ascii')
            activation_functions = metadata["af"].split(",")
            return NeuralNetwork(layers, activation_functions)
        except Exception:
            raise NeuralNetwork.ImportFailedException("Data file corrupted")

    class ImportFailedException(Exception):
        pass

    class RemovalFailedException(Exception):
        pass
    
    class InitializationException(Exception):
        pass

class LGNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, activation_functions=("relu", "sigmoid")):
        if len(layers) < 2:
            raise self.InitializationException("A feedforward neural network has to have at least 2 layers")
        generated_layers = []
        generated_layers.append(InputLayer(layers[0]))
        if len(layers) > 2:
            for l in range(1, len(layers)-1):
                generated_layers.append(HiddenLayer(layers[l], layers[l-1]))
        generated_layers.append(OutputLayer(layers[-1], layers[-2]))
        super().__init__(generated_layers, activation_functions)

