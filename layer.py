import numpy as np
from utils import *
from activations import *

class Layer:
	def __init__(self, act, output_size, input_size=None, name=None, weights=None, biases=None):
		self.act = act
		self.biases = biases
		self.weights = weights
		self.input_size = input_size 
		self.output_size = output_size 

		self.lr = None
		self.input = None
		self.momentum = None
		self.gradients = None
		self.next_layer = None
		self.activation = None
		self.delta_error = None

	def init_weights(self):
		# implementation of Xavier init scheme if weights not given already
		if self.weights == None:
			layer_shape = (self.input_size, self.output_size)
			xavier_weights = np.random.normal(loc=0, scale=4/np.average(layer_shape), size=layer_shape)
			self.weights =  xavier_weights

		if self.biases == None:
			self.biases = np.zeros(self.output_size)

	def init_random_map(self, num_classes):
		# (500, 50) - (50, 10)
		self.random_map = np.random.choice([-1,1], (self.output_size, num_classes))


	def forward_pass(self, inp):
		self.input = inp
		dot_prod = np.dot(inp, self.weights) + self.biases
		self.activation = self.act(dot_prod)
		
		return self.activation

	def backward_pass(self, y):
		self.alt_backward_pass(y)
		return 

		if self.act == softmax:
			self.delta_error = self.activation - y

		elif self.act == tanh:
			tanh_deriv = 1 - self.activation**2
			# (100, 15) = (100, 10) @ (15, 10).T 
			self.delta_error = tanh_deriv * (self.next_layer.delta_error @ self.next_layer.weights.T)
			# self.delta_error = tanh_deriv**2 * (np.average(self.activation) - self.activation)**2 * \
			# 								np.sign((np.average(self.activation) - self.activation))

		batch_gradients = (self.input.T @ self.delta_error)/len(self.input)
		self.gradients = (batch_gradients+self.gradients*self.momentum)/(1+self.momentum)
		self.weights -= self.gradients * self.lr


	def alt_backward_pass(self, y):
		if self.act == softmax:
			self.delta_error = self.activation - y

		elif self.act == tanh:
			tanh_deriv = 1 - self.activation**2
			# (100, 15) = (100, 10) @ (15, 10).T 
			# self.delta_error = tanh_deriv * (self.next_layer.delta_error @ self.next_layer.weights.T)
			# (100,50)
			false_target = y @ self.random_map.T
			self.delta_error =  self.activation - false_target

		batch_gradients = (self.input.T @ self.delta_error)/len(self.input)
		self.gradients = (batch_gradients+self.gradients*self.momentum)/(1+self.momentum)
		self.weights -= self.gradients * self.lr
