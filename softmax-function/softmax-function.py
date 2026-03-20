import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x= np.array(x)
    if x.ndim ==1:
        ex = np.exp(x-np.max(x))
        exp_sum = np.sum(ex)
        return ex/exp_sum
    elif x.ndim ==2:
        ex = np.exp(x - np.max(x,axis=1, keepdims=True))
        exp_sum = np.sum(ex,axis=1, keepdims=True)
        return ex/exp_sum
    pass