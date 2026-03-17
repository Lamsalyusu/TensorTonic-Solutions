import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X =np.array(X)
    y= np.array(y)
    x_t = X.T
    products = x_t @ X
    prods2 = x_t @ y
    inverse =np.linalg.inv(x_t @ X)
    w = inverse @ prods2
    return w
    pass