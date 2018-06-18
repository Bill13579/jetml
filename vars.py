import jetm as jm

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

