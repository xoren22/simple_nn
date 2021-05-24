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

	def init_random_map(self, num_classes):
		self.random_map = np.ones((self.output_size, num_classes))
		for i in range(len(self.random_map)-1):
			if num_classes == 2:
				p = [0.5, 0.5]
			else:
				p = 1-np.linspace(-1,1,num_classes)**2
				p /= sum(p) 
			num_classes_left = np.random.choice(np.arange(num_classes), p = p)
			classes_on_left = np.random.choice(np.arange(num_classes), num_classes_left)
			self.random_map[i, classes_on_left] = 0

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


	def alt_backward_pass(self, y):
		self.y = y
		if self.act_function.name == "softmax":
			self.delta_error = self.activations - self.y

		elif self.act_function.name == "tanh":
			false_target = self.y @ self.random_map.T
			tanh_deriv = 1 - self.activations**2
			self.delta_error = tanh_deriv * (self.activations - np.array([-1,1])[y[:,0]][:,None])**3

		else:
			raise Exception("alt_backward_pass is not implemented for this activation yet")

		raw_batch_gradients = (self.input.T @ self.delta_error)/len(self.input)
		
		self.gradients = self.optimiser.get_update(raw_batch_gradients)
		self.weights -= self.gradients


"""
LOGS
implement relu fake gradient with np.maximum(act*target, 0)
try using next layer weights for beter delta_error estimation
try other approaches for gradinet sign technique
"""
