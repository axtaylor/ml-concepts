import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_encoding(y):
    encoded_feature = torch.zeros((y.size(0), torch.max(y).item() + 1), device=device)
    encoded_feature[torch.arange(y.size(0), device=device), y] = 1
    return encoded_feature

def init_network_params(n_hidden, n_classes, n_features):
    weight_1 = torch.randn((n_hidden, n_features), device=device) * torch.sqrt(torch.tensor(2.0 / n_features))
    bias_1 = torch.zeros((n_hidden, 1), device=device)
    
    weight_2 = torch.randn((n_classes, n_hidden), device=device) * torch.sqrt(torch.tensor(2.0 / n_hidden))
    bias_2 = torch.zeros((n_classes, 1), device=device)
    
    return weight_1, bias_1, weight_2, bias_2

def forward_prop(weight_1, bias_1, weight_2, bias_2, X):

    def ReLU(model):
        return torch.maximum(torch.zeros_like(model), model)
    
    def softmax(model):
        e = torch.exp(model - torch.max(model, dim=0, keepdim=True).values)
        return e / torch.sum(e, dim=0, keepdim=True)

    model_layer_1 = weight_1 @ X + bias_1
    activation_layer_1 = ReLU(model_layer_1)

    model_layer_2 = weight_2 @ activation_layer_1 + bias_2
    activation_layer_2 = softmax(model_layer_2)
    
    return model_layer_1, activation_layer_1, model_layer_2, activation_layer_2

def backward_prop(model_layer_1, activation_layer_1,
                  model_layer_2, activation_layer_2,
                  weight_1, weight_2, X, y):

    def dydx_ReLU(model):
        return (model > 0).float()

    m = X.shape[1]
    y_encoded = one_hot_encoding(y).T

    d_model_2 = activation_layer_2 - y_encoded
    d_weight_2 = (1 / m) * (d_model_2 @ activation_layer_1.T)
    d_bias_2 = (1 / m) * torch.sum(d_model_2, dim=1, keepdim=True)

    d_model_1 = (weight_2.T @ d_model_2) * dydx_ReLU(model_layer_1)
    d_weight_1 = (1 / m) * (d_model_1 @ X.T)
    d_bias_1 = (1 / m) * torch.sum(d_model_1, dim=1, keepdim=True)

    return d_weight_1, d_bias_1, d_weight_2, d_bias_2

def update_network_params(weight_1, bias_1, weight_2, bias_2,
                          d_weight_1, d_bias_1, d_weight_2, d_bias_2, lr):
    
    weight_1 -= lr * d_weight_1
    bias_1   -= lr * d_bias_1    
    weight_2 -= lr * d_weight_2  
    bias_2   -= lr * d_bias_2    

    return weight_1, bias_1, weight_2, bias_2

def predict(activation_layer_2):
    return torch.argmax(activation_layer_2, dim=0)

def score(predictions, y):
    return (predictions == y).float().mean().item()

def cross_entropy_loss(activation_layer_2, y):
    m = y.size(0)
    y_encoded = one_hot_encoding(y).T
    return -torch.sum(y_encoded * torch.log(activation_layer_2 + 1e-8)) / m

def gradient_descent(X, y, n_hidden=None, lr=0.1, n_iters=100):

    X = X.to(device)
    y = y.to(device)
    n_hidden = (torch.unique(y).numel() if n_hidden is None else n_hidden)

    weight_1, bias_1, weight_2, bias_2 = init_network_params(
        n_hidden=n_hidden,
        n_classes=torch.unique(y).numel(),
        n_features=X.shape[0]
    )

    training_log = {'train_loss': [], 'train_acc': []}

    for i in range(n_iters):
        model_1, activation_1, model_2, activation_2 = forward_prop(weight_1, bias_1, weight_2, bias_2, X)
        training_loss = cross_entropy_loss(activation_2, y)

        d_weight_1, d_bias_1, d_weight_2, d_bias_2 = backward_prop(
            model_1, activation_1, model_2, activation_2, weight_1, weight_2, X, y
        )

        weight_1, bias_1, weight_2, bias_2 = update_network_params(
            weight_1, bias_1, weight_2, bias_2,
            d_weight_1, d_bias_1, d_weight_2, d_bias_2, lr
        )

        if i % 100 == 0:
            training_score = score(predict(activation_2), y)
            training_log['train_loss'].append(training_loss.item())
            training_log['train_acc'].append(training_score)
            print(f"Iteration {i:4d} | Loss: {training_loss:.4f} | Train Score: {training_score:.4f}")

    return weight_1, bias_1, weight_2, bias_2, training_log
