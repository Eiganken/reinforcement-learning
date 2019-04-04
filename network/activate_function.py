import numpy as np

class Logistic:
	def __init__(self, alpha=1):
		self.alpha = alpha

	def f(self, x):
		x[x < -700] = -700
		return 1 / (1 + np.exp(-x/self.alpha))

	def df(self, x):
		return self.f(x) * (1 - self.f(x)) / self.alpha


class ReLU:
	def __init__(self):
		self.mask = None

	def f(self, x):
		self.mask = (x <= 0)
		return np.maximum(x, 0)

	def df(self, x):
		x = np.ones_like(x)
		x[self.mask] = 0
		return x


class Linear:
	def f(self, x):
		return x

	def df(self, x):
		return np.ones_like(x)


class Softmax:
	def f(self, x):
		m = np.max(x)
		return np.exp(x-m) / np.sum(np.exp(x-m))