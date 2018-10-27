"""
Simply loads DNA sequences into binary vectors.

Suppose 1 sequence per line.

AACTG --> 0, 0, 1, 0, 1
CTGAT --> 1, 0, 1, 0, 0
"""
import os

import numpy as np
from sklearn import model_selection


WEAK = 'A', 'T'
STRONG = 'C', 'G'
BASE_TO_INT = dict(A=0, T=0, C=1, G=1)


def load_dna_file(filepath):
    """
    Load a string of dna from a text file and returns a list of 0, 1 based on `BASE_TO_INT`.
    """
    vectors = []
    with open(filepath) as fp:
        for seq in fp:
            vector = [BASE_TO_INT[base] for base in seq.strip()]
            vectors.append(vector)
    return vectors


def _load_data(filepath1, filepath2):
    """
    Load 2 classes from filepath1 and filepath2 and return splitted examples and labels in train and set
    using train_test_split.
    """
    X_0 = load_dna_file(filepath1)
    X_1 = load_dna_file(filepath2)
    X = np.array(X_0 + X_1)
    y_0 = [0] * len(X_0)
    y_1 = [1] * len(X_1)
    y = np.array(y_0 + y_1)
    return model_selection.train_test_split(X, y)


def load_data(dirpath):
    """
    Load the 2 classes of sequences to classify and return (X_train, y_train), (X_test, y_test)
    just like minst.load_data().
    """
    f1 = os.path.expanduser(os.path.join(dirpath, 'non-promoters_upstream.dna'))
    f2 = os.path.expanduser(os.path.join(dirpath, 'promoters.dna'))
    X_train, X_test, y_train, y_test = _load_data(f1, f2)
    return (X_train, y_train), (X_test, y_test)


def plot(data):
    """Plot the mean of data along axis 0
    """
    # useful to plot the mean base frequency per position
    import matplotlib.pyplot as plt
    plt.plot(data.mean(axis=0))
    plt.show()


if __name__ == '__main__':
    import sys
    X_train, X_test, y_train, y_test = load_data(sys.argv[1])
    print('X_train.shape =', X_train.shape)
    print('X_test.shape =', X_test.shape)
    print('y_train.shape =', y_train.shape)
    print('y_test.shape =', y_test.shape)
