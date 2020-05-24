import pickle
import numpy as np
import matplotlib.pyplot as plt

from nn import Net
from utils import *
from layer import Layer
from optimisers import *
from activations import *

import keras
from keras.datasets import mnist
from keras.datasets import cifar10



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train), -1)/255
x_test  =  x_test.reshape(len(x_test),  -1)/255


num_classes = 10
classes = np.random.choice(np.arange(10), num_classes, replace=False)

inds_train = [y in classes for y in y_train]
inds_test  = [y in classes for y in y_test]

y_train, y_test = y_train.flatten(), y_test.flatten()
x_train, x_test = x_train[inds_train], x_test[inds_test]

y_test  = to_categorical(y_test[inds_test]).astype(np.int)
y_train = to_categorical(y_train[inds_train]).astype(np.int)

net = Net()
net.add_layer(Layer(act=tanh, output_size=50, input_size=784))
net.add_layer(Layer(act=tanh, output_size=50))
net.add_layer(Layer(act=tanh, output_size=50))
net.add_layer(Layer(act=softmax, output_size=num_classes))

net.compile(optimiser=Adam())

print('\n'*3,'-'*100, '\n'*3)
bug_print(((x_train.shape, y_train.shape), (x_test.shape, y_test.shape)))
print('\n'*2,'-'*100, '\n'*2)


# y_logs = np.zeros((2000,1000, num_classes))
# activation_logs = np.zeros((2000,1000,50))
# delta_error_logs = np.zeros((2000,1000,50))

for i in range(40):
	inds = np.random.choice(np.arange(len(x_train)), 128)
	x_batch, y_batch = x_train[inds], y_train[inds]

	net.forward_pass(x_batch)
	net.backward_pass(y_batch)

	# if i % 5 == 0:
	# 	activation_logs[i//5] = net.layers[3].activations
	# 	delta_error_logs[i//5] = net.layers[3].delta_error
	# 	y_logs[i//5] = y_batch



	if i % 100 == 0:
		test_acc, test_loss, train_ginni = net.metrics(x_test, y_test)	
		train_acc, train_loss, test_ginni = net.metrics(x_train[:10000], y_train[:10000])	

		print(i, train_acc, test_acc, train_loss, test_loss, sep='\t\t\t')

# np.save('./logs/y_logs', y_logs)
# np.save('./logs/activation_logs', activation_logs)
# np.save('./logs/delta_error_logs', delta_error_logs)
