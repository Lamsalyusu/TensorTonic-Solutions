import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    # find most frequent label
    values, counts = np.unique(y_train, return_counts=True)
    majority_label = values[np.argmax(counts)]

    # predict for all test samples
    predictions = np.full(X_test.shape[0], majority_label)

    return predictions
    pass