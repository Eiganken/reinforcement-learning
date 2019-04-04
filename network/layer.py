import numpy as np

class Layer:
	def __init__(self, size, activate_function, optimizer, loss=None):
		self.W = np.random.randn(*size)
		self.b = np.random.randn(size[1])
		self.function = activate_function
		self.optimizer = optimizer
		self.loss_function = loss

		self.u, self.z, self.delta = None, None, None

		self.dW, self.db = 0, 0

		self.loss = None

	def forward(self, z):
		self.z = z.reshape((-1,1))
		self.u = np.dot(z, self.W) + self.b
		return self.function.f(self.u)

	def backpropagation(self, delta=None, W=None, y=None, d=None):
		if y is not None and d is not None:
			self.loss = self.loss_function.f(y, d)
			self.delta = self.loss * self.loss_function.df(y, d)
		else:
			self.delta = self.function.df(self.u) * np.dot(W, delta.T)
		return self.delta, self.W

	def update(self, batch, batch_size):
		self.dW += np.dot(self.z, self.delta.reshape((1,-1)))
		self.db += self.delta

		if batch:
			self.W, self.b = self.optimizer(self.W, self.b, self.dW, self.db, batch_size)
			self.dW, self.db = 0, 0

	def copy(self):
		layer = Layer(self.W.shape, self.function, self.optimizer, self.loss_function)
		layer.W = self.W.copy()
		layer.b = self.b.copy()
		return layer