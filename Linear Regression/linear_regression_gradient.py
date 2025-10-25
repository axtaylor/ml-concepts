import torch
import numpy as np

class LinearRegressionGradient:

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000, device = None) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate, self.n_iters = learning_rate, n_iters
        self.coefficients, self.intercept = None, None

    def fit(self, X, y):
        #X = torch.tensor(X, dtype=torch.float32, device=self.device)
        #y = torch.tensor(y, dtype=torch.float32, device=self.device)

        n, k = X.shape   
        self.coefficients = torch.randn(k, device=self.device, dtype=torch.float32, requires_grad=False)
        self.intercept = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False)

        for _ in range(self.n_iters): 
            y_pred = X @ self.coefficients + self.intercept
            gradient_weights = (1 / n) * (X.T @ (y_pred - y))
            gradient_wrt_bias = (1 / n) * torch.sum(y_pred - y)
            self.coefficients -= self.learning_rate * gradient_weights
            self.intercept -= self.learning_rate * gradient_wrt_bias

        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return (X @ self.coefficients + self.intercept).cpu().detach().numpy()
    
