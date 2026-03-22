import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y= np.array(y)
    #count occurrences of each class
    values , counts = np.unique(y,return_counts = True)
    # probabilities
    probs = counts / len(y)
    
    probs = probs[probs>0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
    pass