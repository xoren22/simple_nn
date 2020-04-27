import numpy as np
import matplotlib.pyplot as plt

import keras
import numpy as np
import livelossplot
from keras.losses import *
from keras.layers import *
from keras.optimizers import *
from keras import activations
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from IPython.display import clear_output
from keras.utils import to_categorical as sparse_categorical



def tanh(arr):
    return np.tanh(arr)

def softmax(arr):
    exp = np.exp(np.asarray(arr))
    return exp / exp.sum(1)[:, None]

def to_categorical(y):
    uniques = np.unique(y)
    sparse_cat = np.zeros((len(y), np.max(y)+1))
    sparse_cat[np.arange(len(y)), y] = True
    res = sparse_cat[:, uniques]
    
    return res

def bug_print(string):
	print("*"*40)
	print("*"*40)
	print("\n"*2)
	print(string)
	print("\n"*2)
	print("*"*40)
	print("*"*40)



class Layer:
	def __init__(self, act, output_size, name=None, input_size=None, weights=None, biases=None):
		self.act = act
		self.biases = biases
		self.weights = weights
		self.input_size = input_size 
		self.output_size = output_size 

		self.next_layer = None
		self.last_activation = None
		self.last_delta_error = None

	def init_weights(self):
		# implementation of Xavier init scheme if weights not given already
		if self.weights == None:
			layer_shape = (self.input_size, self.output_size)
			xavier_weights = np.random.normal(loc=0, scale=1/np.average(layer_shape), size=layer_shape)
			self.weights =  xavier_weights

		if self.biases == None:
			self.biases = np.zeros(self.output_size)

	def forward_pass(self, inp):
		dot_prod = np.dot(inp, self.weights) + self.biases
		self.last_activation = self.act(dot_prod)
		return self.last_activation

	def update_weights(self):
		NotImplementedError





class Net:
	def __init__(self):
		self.layers = []
		self.num_layers = 0

	def add_layer(self, layer):
		self.num_layers += 1
		if layer.input_size == None:
			layer.input_size = self.layers[-1].output_size

		layer.name = "layer_%d" % self.num_layers
		layer.init_weights()
		self.layers.append(layer)

	def set_weigts(self, W):
		weights, biases = W[::2], W[1::2]

		for i in range(self.num_layers):
			new_weights_shape = W[2*i].shape
			new_biases_shape  = W[2*i+1].shape

			old_weights_shape = (self.layers[i].input_size, self.layers[i].output_size)
			old_biases_shape  = (self.layers[i].output_size,)

			assert old_weights_shape == new_weights_shape, "weight shape mismatch on layer %s" % self.layers[i].name
			assert old_biases_shape  == new_biases_shape,  "biases shape mismatch on layer %s" % self.layers[i].name

			self.layers[i].weights = W[2*i]
			self.layers[i].biases  = W[2*i+1]

	def forward_pass(self, inp):
		lyr_outs = inp
		for lyr in self.layers:
			lyr_outs = lyr.forward_pass(lyr_outs)

		return lyr_outs

	def backward_pass(self):
		for lyr in self.layers:
			info = lyr.backward_pass(info)

	def predict(self, inp):
		scores = self.forward_pass(inp)
		return scores.argmax(1)



if __name__ == "__main__":
	net = Net()
	net.add_layer(Layer(act=tanh, output_size=10, input_size=3))
	net.add_layer(Layer(act=tanh, output_size=10))
	net.add_layer(Layer(act=tanh, output_size=10))
	net.add_layer(Layer(act=softmax, output_size=2))

	model = Sequential()
	act = activations.tanh
	model.add(Dense(input_shape = (3,), units=10, activation=act))
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(2, activation='softmax'))

	opt = Nadam(2e-4)
	model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])


	bug_print("setting the weights")
	W = model.get_weights()
	net.set_weigts(W)

	inp = np.random.normal(0,1,(4,3))
	print(model.predict(inp)-net.forward_pass(inp))


