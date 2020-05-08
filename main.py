import numpy as np
import matplotlib.pyplot as plt

from nn import Net
from utils import *
from layer import Layer
from activations import *

import keras
from keras.datasets import mnist


net = Net()
net.add_layer(Layer(act=tanh, output_size=50, input_size=784))
net.add_layer(Layer(act=tanh, output_size=50))
net.add_layer(Layer(act=tanh, output_size=50))
net.add_layer(Layer(act=softmax, output_size=2))

net.compile(lr=0.01, momentum=0.9)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)/255
x_test  = x_test.reshape (10000, 784)/255


classes = (7,1)

inds_train = [y in classes for y in y_train]
inds_test  = [y in classes for y in y_test]

x_train = x_train[inds_train]
x_test  = x_test[inds_test]

y_train = to_categorical(y_train[inds_train]).astype(np.int)
y_test  = to_categorical(y_test[inds_test]).astype(np.int)

print('\n'*3,'-'*100, '\n'*3)

print((x_train.shape, y_train.shape), (x_test.shape, y_test.shape))

print('\n'*2,'-'*100, '\n'*2)

for i in range(1000):
	inds = np.random.choice(np.arange(len(x_train)), 500)
	x_batch, y_batch = x_train[inds], y_train[inds]

	net.forward_pass(x_batch)
	net.backward_pass(y_batch)

	if i % 100 == 0:
		acc, loss = net.metrics(x_test, y_test)	
		print(i, acc, loss, sep='\t\t\t')


