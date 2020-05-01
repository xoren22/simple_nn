from nn import Net, Layer
from activations import *
from utils import bug_print

import numpy as np
from keras.layers import Dense
from keras.models import Sequential



def test_set_weights():
	net = Net()
	net.add_layer(Layer(act=tanh, output_size=10, input_size=3))
	net.add_layer(Layer(act=tanh, output_size=10))
	net.add_layer(Layer(act=tanh, output_size=10))
	net.add_layer(Layer(act=softmax, output_size=2))

	model = Sequential()
	model.add(Dense(input_shape = (3,), units=10, activation="tanh"))
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(2, activation='softmax'))

	model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

	W = model.get_weights()
	net.set_weigts(W)

	print("Successfully set the weights from keras model")
	return net, model

def test_forward_pass():
	my_net, keras_net = test_set_weights()

	inp = np.random.normal(0,1,(4,3))
	my_model_pred, keras_model_pred = my_net.forward_pass(inp), keras_net.predict(inp)

	avg_abs_diff = np.average(np.abs(my_model_pred-keras_model_pred))
	
	return avg_abs_diff



if __name__ == "__main__":
	test_set_weights()
	res = test_forward_pass()
	bug_print("Average absolute difference between model predicts - %f16"%res)
