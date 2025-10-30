import numpy as np

def one_hot_encoding(y):
    encoded_feature = np.zeros((y.size, np.max(y) + 1))
    encoded_feature[np.arange(y.size), y] = 1
    return encoded_feature

def init_network_params(n_hidden, n_classes, n_features):

    weight_1 = np.random.randn(n_hidden, n_features) * np.sqrt(2.0 / n_features)
    bias_1 = np.zeros((n_hidden, 1))
    
    weight_2 = np.random.randn(n_classes, n_hidden) * np.sqrt(2.0 / n_hidden)
    bias_2 = np.zeros((n_classes, 1))
    
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
        e = np.exp(model - np.max(model, axis=0, keepdims=True))
        return e / np.sum(e, axis=0, keepdims=True)

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

def backward_prop(model_layer_1, activation_layer_1,
        model_layer_2, activation_layer_2,
        weight_1, weight_2,
        X, y,
    ):

    def dydx_ReLU(model):
        return model > 0

    m = X.shape[1]
    y_encoded = one_hot_encoding(y).T

    d_model_2 = activation_layer_2 - y_encoded
    d_weight_2 = 1 / m * d_model_2.dot(activation_layer_1.T)
    d_bias_2 = 1 / m * np.sum(d_model_2, axis=1, keepdims=True)

    d_model_1 = weight_2.T.dot(d_model_2) * dydx_ReLU(model_layer_1)
    d_weight_1 = 1 / m * d_model_1.dot(X.T)
    d_bias_1 =  1/ m * np.sum(d_model_1, axis=1, keepdims=True)

    return (
        d_weight_1,
        d_bias_1,
        d_weight_2,
        d_bias_2
)

def update_network_params(weight_1, bias_1, weight_2, bias_2,
                          d_weight_1, d_bias_1, d_weight_2, d_bias_2,
                          lr,
                        ):
    
    weight_1 -= lr * d_weight_1
    bias_1   -= lr * d_bias_1    
    weight_2 -= lr * d_weight_2  
    bias_2   -= lr * d_bias_2    

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

def cross_entropy_loss(activation_layer_2, y):
    m = y.size
    y_encoded = one_hot_encoding(y).T
    return -np.sum(y_encoded * np.log(activation_layer_2 + 1e-8)) / m

def gradient_descent(X, y, n_hidden=None, lr=0.1, n_iters=100):

    # Default hidden layers to output layer size unless specified.
    n_hidden = (np.unique(y).size if n_hidden is None else n_hidden)

    weight_1, bias_1, weight_2, bias_2 = init_network_params(n_hidden=n_hidden, n_classes=np.unique(y).size, n_features=X.shape[0])

    training_log = {'train_loss': [], 'train_acc': []}
    for i in range(n_iters):

        model_1, activation_1, model_2, activation_2 = forward_prop(weight_1, bias_1, weight_2, bias_2, X)
        training_loss = cross_entropy_loss(activation_2, y)

        d_weight_1, d_bias_1, d_weight_2, d_bias_2 = backward_prop(model_1, activation_1, model_2, activation_2, weight_1, weight_2, X, y)
        weight_1, bias_1, weight_2, bias_2 = update_network_params(weight_1, bias_1, weight_2, bias_2, d_weight_1, d_bias_1, d_weight_2, d_bias_2, lr)

        if i % 10 == 0:
            training_score = score(predict(activation_2), y)
            training_log['train_loss'].append(training_loss)
            training_log['train_acc'].append(training_score)
            print(f"Iteration {i:4d} | Loss: {training_loss:.4f} | Train Score: {training_score:.4f}\n")

    return (
        weight_1,
        bias_1,
        weight_2,
        bias_2,
        training_log
)


