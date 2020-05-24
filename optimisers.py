import numpy as np

class SGD:
	"""docstring for SGD"""
	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.gradients = 0
		self.momentum = momentum


	def get_update(self, new_batch_gradients):
		self.gradients = self.lr*(new_batch_gradients+self.gradients*self.momentum)/(1+self.momentum)
		return self.gradients

class Adam:
	def __init__(self, lr=0.001, beta1 = 0.9, beta2 = 0.999, eps_stable = 1e-8):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps_stable = eps_stable

		self.t = 0     			# iteration counter
		self.m = 0			# first moment vector
		self.v = 0			# second moment vector
		self.biased_m = 0	# biased first moment vector
		self.biased_v = 0	# biased second moment vector

	def get_update(self, new_batch_gradients):
		self.t += 1
		self.biased_m = self.beta1 * self.biased_m + (1-self.beta1)*new_batch_gradients
		self.biased_v = self.beta2 * self.biased_v + (1-self.beta2)*new_batch_gradients**2

		self.m = self.biased_m / (1-self.beta1)
		self.v = self.biased_v / (1-self.beta2)

		update = self.lr * self.m / (np.sqrt(self.v)+self.eps_stable)

		return update




