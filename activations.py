import numpy as np

class tanh:
	def __init__(self):
		self.name = "tanh"

	def forward(self, inp):
		return np.tanh(inp)
	def derivative(self, inp):
		return 1 - np.square(inp)

class softmax:
	def __init__(self):
		self.name = "softmax"

	def forward(self, inp):
		exp = np.exp(np.asarray(inp))
		return exp / exp.sum(1)[:, None]

	def derivative(self, inp, y):
		return inp - y

class relu:
	def __init__(self):
		self.name = "relu"

	def forward(self, inp):
		return np.maximum(inp, 0)

	def derivative(self, inp):
		return inp > 0

def sigmoid(inp, tmp=1):
	return 1/(1 + np.exp(-inp*tmp)) 
