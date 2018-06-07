import numpy as np

LEAKY_RELU_A = 0.01

def identity(x):
    return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return max(0,x)

def leaky_relu(x):
    return x if x > 0 else LEAKY_RELU_A*x

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu_p(x):
    return 1 if x > 0 else 0

def leaky_relu_p(x):
    return 1 if x > 0 else LEAKY_RELU_A

def multiply_combinations(x, y):
    matrix = []
    for i in range(x.shape[0]):
        matrix.append([])
        for j in range(y.shape[0]):
            matrix[i].append(x[i, 0] * y[j, 0])
    return np.matrix(matrix)

def multiply_across(m, v):
    if m.shape[0] != v.shape[0]:
        raise Exception("Matrix and vector rows count do not match")
    matrix = []
    for i in range(v.shape[0]):
        matrix.append([])
        for j in range(m.shape[1]):
            matrix[i].append(m[i, j] * v[i, 0])
    return np.matrix(matrix)

