import numpy as np

class Layer:
    def __init__(self, dimensionality, last_layer_dimensionality=None, weights=None, biases=None, no_biases=False):
        self.dimensionality = dimensionality
        if weights is not None:
            self.weights = weights
        else:
            if last_layer_dimensionality is not None:
                self.weights = np.matrix(np.random.randn(self.dimensionality, last_layer_dimensionality))
            else:
                self.weights = None
        if biases is not None:
            self.biases = biases
        else:
            if not no_biases:
                self.biases = np.matrix(np.random.randn(self.dimensionality, 1))
            else:
                self.biases = None
    
    class InitializationException(Exception):
        pass

class InputLayer(Layer):
    def __init__(self, dimensionality):
        super().__init__(dimensionality, no_biases=True)

class HiddenLayer(Layer):
    def __init__(self, dimensionality, last_layer_dimensionality=None, weights=None, biases=None):
        super().__init__(dimensionality, last_layer_dimensionality, weights, biases)

class OutputLayer(HiddenLayer):
    pass

