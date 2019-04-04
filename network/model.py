import matplotlib.pyplot as plt
import numpy as np
import copy

class NeuralNetwork:
	def __init__(self, *layers, batch_size=50):
		self.layers = layers

		self.batch_size = batch_size
		self.batch_counter = 0

	def predict(self, x):
		z = x
		for layer in self.layers:
			z = layer.forward(z)
		return z

	def evaluate(self, y, d):
		for i, layer in enumerate(self.layers[::-1]):
			if i == 0:
				delta, W = layer.backpropagation(y=y, d=d)
			else:
				delta, W = layer.backpropagation(delta=delta, W=W)
		return self.layers[-1].loss

	def learn(self):
		self.batch_counter += 1
		for layer in self.layers:
			layer.update(self.batch_counter==self.batch_size, self.batch_size)
		if self.batch_counter == self.batch_size:
			self.batch_counter = 0

	def fit(self, x_train, t_train):
		loss = 0
		for x, t in zip(x_train, t_train):
			y = self.predict(x)
			loss += self.evaluate(y, t)
			self.learn()
		return loss / x_train.shape[0]

	def copy(self):
		layers = []
		for layer in self.layers:
			layers.append(layer.copy())
		network = NeuralNetwork(*layers, batch_size=self.batch_size)
		return network