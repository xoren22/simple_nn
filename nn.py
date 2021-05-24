import numpy as np
from utils import *
from layer import Layer
from optimisers import *
from activations import *
from copy import deepcopy

class Net:
	def __init__(self):
		self.layers = []
		self.num_layers = 0

	def add_layer(self, layer):
		self.num_layers += 1
		if layer.input_size == None:
			layer.input_size = self.layers[-1].output_size

		layer.name = "layer_%d" % self.num_layers
		self.layers.append(layer)

	def set_weigts(self, W):
		weights, biases = W[::2], W[1::2]

		for i in range(self.num_layers):
			new_weights_shape = W[2*i].shape
			new_biases_shape  = W[2*i+1].shape

			old_weights_shape = (self.layers[i].input_size, self.layers[i].output_size)
			old_biases_shape  = (self.layers[i].output_size,)

			assert old_weights_shape == new_weights_shape, \
										"weight shape mismatch on layer %s" % self.layers[i].name
			assert old_biases_shape  == new_biases_shape,  \
										"biases shape mismatch on layer %s" % self.layers[i].name

			self.layers[i].weights = W[2*i]
			self.layers[i].biases  = W[2*i+1]

	def get_weights(self, lyrs=None):
		if lyrs == None:
			lyrs = np.arange(self.num_layers)

		weights = []
		for i in lyrs:
			weights.append(self.layers[i].weights)
			weights.append(self.layers[i].biases)

		return weights

	def get_activations(self, lyrs=None):
		if lyrs == None:
			lyrs = np.arange(self.num_layers)

		activation = [self.layers[i].activations for i in lyrs]

		return activation



	def compile(self, optimiser=SGD):
		for i, lyr in enumerate(self.layers):
			lyr.optimiser = deepcopy(optimiser)

			if lyr.weights is None:
				lyr.init_weights()
				lyr.init_random_map(self.layers[-1].output_size)
				lyr.gradients = np.zeros(lyr.weights.shape)

				lyr.next_layer = self.layers[i+1] if i != self.num_layers-1 else None


	def accurecy(self, x, y):
		class_number = np.where(y)[1]

		y_pred = np.argmax(self.forward_pass(x), 1)
		y_true = np.argmax(y, 1)

		return np.average(y_pred == y)

	def metrics(self, x, y):
		y_prob = self.forward_pass(x)
		y_pred = np.argmax(y_prob, 1)
		y_true = np.argmax(y, 1)

		acc =  np.average(y_pred == y_true)
		loss = -1*np.average(np.log(y_prob[y.astype(bool)]))
		ginni = [lyr.ginni() for lyr in self.layers]

		return acc, loss, ginni


	def forward_pass(self, inp):
		lyr_outs = inp
		for lyr in self.layers:
			lyr_outs = lyr.forward_pass(lyr_outs)

		return lyr_outs

	def backward_pass(self, y):
		for lyr in self.layers[::-1]:
			lyr.backward_pass(y)

	# def alt_backward_pass(self, y):
	# 	for lyr in self.layers[::-1]:
	# 		lyr.alt_backward_pass(y)


	def predict(self, inp):
		scores = self.forward_pass(inp)
		return scores.argmax(1)




if __name__ == "__main__":
	net = Net()
	net.add_layer(Layer(act=tanh, output_size=100, input_size=784))
	net.add_layer(Layer(act=tanh, output_size=100))
	net.add_layer(Layer(act=tanh, output_size=10))
	net.add_layer(Layer(act=softmax, output_size=2))

	net.compile(optimiser=SGD(0.01, 0.9))

	x = np.random.uniform(0,1,(100, 784))
	y = np.random.randint(0,2,(100, 2))
	net.forward_pass(x)
	net.backward_pass(y)

	net.get_activations()
	net.get_weights()

	bug_print("No exceptions on random data!")






