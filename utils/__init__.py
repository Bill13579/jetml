import numpy as np
import copy

def array_to_vector(x):
    return np.matrix(x).reshape((len(x), 1))

