import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    X = np.array(X)
    y = np.array(y)
    b = 0.0
    n_samples = X.shape[0]
    # w = np.zeros(n_samples)
    w = np.zeros(X.shape[1])
    b = 0.0
    # Write code here
    for _ in range (steps):
        z = X @ w +b
        y_pred = _sigmoid(z)
        error = y_pred - y 
        dw = (1/n_samples) * (X.T @ error)
        db = (1/n_samples) * np.sum(error)

        w -= lr *dw
        b -= lr*db
    return w,b
    pass