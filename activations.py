import numpy as np

def tanh(arr):
    return np.tanh(arr)

def softmax(arr):
    exp = np.exp(np.asarray(arr))
    return exp / exp.sum(1)[:, None]

def sigmoid(arr, temp=10):
	return 1/(1 + np.exp(-arr*temp)) 

def ralu(arr):
	return np.maximum(arr, 0)