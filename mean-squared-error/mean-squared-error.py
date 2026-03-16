import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
     """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    error = np.mean((y_pred-y_true)**2)
    return error
    
    # y_true = [1,2,3,4,5]
    # y_pred = [1.1,2.1,2.9,4.2,4.8]

    # print(mean_squared_error(y_pred, y_true))
    pass
    
    
