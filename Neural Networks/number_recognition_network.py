import numpy as np
import matplotlib.pyplot as plt

def one_hot_encoding(y):
    encoded_feature = np.zeros((y.size, np.max(y) + 1))
    encoded_feature[np.arange(y.size), y] = 1
    return encoded_feature

def init_network_params(n_neurons, n_features):

    weight_1 = np.random.rand(n_neurons, n_features)-0.5
    bias_1 = np.random.rand(n_neurons, 1)-0.5
    
    weight_2 = np.random.rand(n_neurons, n_neurons)-0.5
    bias_2 = np.random.rand(n_neurons, 1)-0.5
    
    return (
        weight_1,
        bias_1,
        weight_2,
        bias_2,
)

def forward_prop(weight_1, bias_1, weight_2, bias_2, X):

    def ReLU(model):
        return np.maximum(0, model)
    
    def softmax(model):
        result = np.exp(model) / sum(np.exp(model))
        return result

    model_layer_1 = weight_1.dot(X) + bias_1
    activation_layer_1 = ReLU(model_layer_1)

    model_layer_2 = weight_2.dot(activation_layer_1) + bias_2
    activation_layer_2 = softmax(model_layer_2)
    
    return (
        model_layer_1,
        activation_layer_1,
        model_layer_2,
        activation_layer_2
)

def backward_prop(
        model_layer_1,
        activation_layer_1,
        model_layer_2,
        activation_layer_2,
        weight_1,
        weight_2,
        X,
        y,
        m,
    ):

    def dydx_ReLU(model):
        return model > 0

    y_encoded = one_hot_encoding(y).T

    d_model_2 = activation_layer_2 - y_encoded
    d_weight_2 = 1 / m * d_model_2.dot(activation_layer_1.T)
    d_bias_2 = 1 / m * np.sum(d_model_2)

    d_model_1 = weight_2.T.dot(d_model_2) * dydx_ReLU(model_layer_1)
    d_weight_1 = 1 / m * d_model_1.dot(X.T)
    d_bias_1 =  1/ m * np.sum(d_model_1, axis=1, keepdims=True)

    return (
        d_weight_1,
        d_bias_1,
        d_weight_2,
        d_bias_2
)

def update_network_params(weight_1,
                          bias_1,
                          weight_2,
                          bias_2,
                          d_weight_1,
                          d_bias_1,
                          d_weight_2,
                          d_bias_2,
                          alpha
    ):
    
    weight_1 = weight_1 - alpha * d_weight_1
    bias_1 = bias_1 - alpha * d_bias_1    
    weight_2 = weight_2 - alpha * d_weight_2  
    bias_2 = bias_2 - alpha * d_bias_2    

    return (
        weight_1,
        bias_1,
        weight_2,
        bias_2
)

def predict(activation_layer_2):
    return np.argmax(activation_layer_2, 0)

def score(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(X, y, alpha, n_iters):

    weight_1, bias_1, weight_2, bias_2 = init_network_params(n_neurons=np.unique(y).size, n_features=X.shape[0])

    for i in range(n_iters):

        model_1, activation_1, model_2, activation_2 = forward_prop(weight_1, bias_1, weight_2, bias_2, X)
        d_weight_1, d_bias_1, d_weight_2, d_bias_2 = backward_prop(model_1, activation_1, model_2, activation_2, weight_1, weight_2, X, y, X.shape[1])
        weight_1, bias_1, weight_2, bias_2 = update_network_params(weight_1, bias_1, weight_2, bias_2, d_weight_1, d_bias_1, d_weight_2, d_bias_2, alpha)

        if i % 25 == 0:
            print("Iteration: ", i)
            print(score(predict(activation_2), y))

    return (
        weight_1,
        bias_1,
        weight_2,
        bias_2
)


