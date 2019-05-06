import numpy as np

class Sigmoid:
	def __init__(self, alpha=1e-01):
		self.alpha = alpha

	def f(self, x):
		x[x < -700] = -700
		y = 1 / (1 + np.exp(-self.alpha*x))

		return y

	def df(self, x):
		x_new = self.alpha * x
		dy = self.alpha * self.f(x) * (1-self.f(x))

		return dy

class Linear:
	def f(self, x):
		return x

	def df(self, x):
		return np.ones_like(x)

class ReLU:
	def __init__(self):
		self.mask = None

	def f(self, x):
		y = np.maximum(0, x)

		return y

	def df(self, x):
		dy = np.ones_like(x)
		dy = np.where(dy <= 0, 0, dy)

		return dy


class Softmax:
	def __init__(self):
		self.alpha = 1e-01

	def f(self, x):
		x_new = x-np.max(x)

		return np.exp(x_new) / np.sum(np.exp(x_new))


class R2error:
	def __init__(self):
		pass
	
	def f(self, y, d):
		loss = y - d
		y =  (loss@loss) / 2

		return y

	def df(self, y, d):
		dy = y - d

		return dy


class Huberloss:
	def __init__(self, ipusironn=0.25):
		self.ipusironn = ipusironn

	def f(self, y, d):
		loss = y - d
		L1 = loss**2 * 0.5
		L2 = self.ipusironn * (np.abs(loss) - self.ipusironn*0.5)
		loss = np.where(np.abs(loss) <= self.ipusironn, L1, L2)
		loss = np.sum(loss)

		return loss

	def df(self, y, d):
		delta = y - d
		delta = np.where(delta > self.ipusironn*0.5, self.ipusironn, delta)
		delta = np.where(delta < -self.ipusironn*0.5, -self.ipusironn, delta)
		return delta