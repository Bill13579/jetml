import jetmath as jm

ACTIVATION_FUNCTIONS = {
    "relu": {
        "_": jm.nonlin.relu,
        "derivative": jm.nonlin.relu_p
    },
    "leaky-relu": {
        "_": jm.nonlin.leaky_relu,
        "derivative": jm.nonlin.leaky_relu_p
    },
    "sigmoid": {
        "_": jm.nonlin.sigmoid,
        "derivative": jm.nonlin.sigmoid_p
    }
}

