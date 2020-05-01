import numpy as np

def tanh(arr):
    return np.tanh(arr)

def softmax(arr):
    exp = np.exp(np.asarray(arr))
    return exp / exp.sum(1)[:, None]

def ralu(arr):
	return np.maximum(arr, 0)