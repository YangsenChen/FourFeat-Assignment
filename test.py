import numpy as np

from models.neural_net import NeuralNetwork



def rel_error(x, y):
    """Returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


input_size = 2
hidden_size = 10
num_classes = 3
num_inputs = 5
optimizer = 'SGD'


def init_toy_model(num_layers):
    """Initializes a toy model"""
    np.random.seed(0)
    hidden_sizes = [hidden_size] * (num_layers - 1)
    return NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers)

def init_toy_data():
    """Initializes a toy dataset"""
    np.random.seed(0)
    X = np.random.randn(num_inputs, input_size)
    y = np.random.randn(num_inputs, num_classes)

    # y = np.random.randint(num_classes, size=num_inputs)

    return X, y

from copy import deepcopy

from utils.gradient_check import eval_numerical_gradient


X, y = init_toy_data()


def f(W):
    net.forward(X)
    return net.backward(y)

for num in [2, 3]:
    net = init_toy_model(num)
    net.forward(X)
    net.backward(y)
    gradients = deepcopy(net.gradients)

    for param_name in net.params:
        param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, gradients[param_name])))