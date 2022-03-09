import numpy as np
from utils import *
from activations import *

class Layer:
	def __init__(self, act, output_size, input_size=None, name=None, weights=None, biases=None):
		self.biases = biases
		self.weights = weights
		self.act_function = act()
		self.input_size = input_size 
		self.output_size = output_size 

		self.y = None
		self.lr = None
		self.input = None
		self.momentum = None
		self.gradients = None
		self.optimiser = None
		self.next_layer = None
		self.activations = None
		self.delta_error = None


	def init_weights(self):
		# implementation of Xavier init scheme if weights not given already
		layer_shape = (self.input_size, self.output_size)
		xavier_weights = np.random.normal(loc=0, scale=4/np.average(layer_shape), size=layer_shape)
		self.weights =  xavier_weights

		self.biases = np.zeros(self.output_size)

	def forward_pass(self, inp):
		self.input = inp
		dot_prod = np.dot(inp, self.weights) + self.biases
		self.activations = self.act_function.forward(dot_prod)
		
		return self.activations

	def ginni(self):
		return 0

	def backward_pass(self, y):
		# self.alt_backward_pass(y); return 
		self.y = y

		if self.act_function.name == 'softmax':
			self.delta_error = self.activations - self.y

		else:
			act_deriv = self.act_function.derivative(self.activations)
			self.delta_error = (self.next_layer.delta_error @ self.next_layer.weights.T) * act_deriv
				
		raw_batch_gradients = (self.input.T @ self.delta_error)/len(self.input)
		self.gradients = self.optimiser.get_update(raw_batch_gradients)
		self.weights -= self.gradients


"""
LOGS
implement relu fake gradient with np.maximum(act*target, 0)
try using next layer weights for beter delta_error estimation
try other approaches for gradinet sign technique
"""