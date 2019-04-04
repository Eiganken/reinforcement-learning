import numpy as np

class Adam:
	def __init__(self, alpha=0.001, beta_first=0.9, beta_second=0.999, ipusironn=1.0e-8):
		self.alpha = alpha
		self.beta_first = beta_first
		self.beta_second = beta_second
		self.ipusironn = ipusironn

		self.W_m = 0
		self.W_v = 0
		self.b_m = 0
		self.b_v = 0
		self.t = 1

	def __call__(self, W, b, dW, db, batch_size):
		self.W_m =  self.beta_first*self.W_m + (1-self.beta_first)*(dW/batch_size)
		self.W_v =  self.beta_second*self.W_v + (1-self.beta_second)*(dW/batch_size)**2
		self.b_m =  self.beta_first*self.b_m + (1-self.beta_first)*(db/batch_size)
		self.b_v =  self.beta_second*self.b_v + (1-self.beta_second)*(db/batch_size)**2

		W_m = self.W_m / (1 - self.beta_first**self.t)
		W_v = self.W_v / (1 - self.beta_second**self.t)
		b_m = self.b_m / (1 - self.beta_first**self.t)
		b_v = self.b_v / (1 - self.beta_second**self.t)

		W -= self.alpha * W_m / (W_v**0.5 + self.ipusironn)
		b -= self.alpha * b_m / (b_v**0.5 + self.ipusironn)

		self.t += 1

		return W, b

class Momentum:
	def __init__(self, mu=0.7, ipusironn=0.4):
		self.W = None
		self.b = None
		self.mu = mu
		self.ipusironn = ipusironn

	def __call__(self, W, b, dW, db, batch_size):
		if self.W is None and self.b is None:
			del_W, del_b = 0, 0
		else:
			del_W, del_b = W - self.W.copy(), b - self.b.copy()
		self.W, self.b = W.copy(), b.copy()
		W -= self.ipusironn * (dW/batch_size) + self.mu*del_W
		b -= self.ipusironn * (db/batch_size) + self.mu*del_b
		return W, b


class Gradient:
	def __init__(self, ipusironn=0.45):
		self.ipusironn = ipusironn

	def __call__(self, W, b, dW, db, batch_size):
		W -= self.ipusironn*dW / batch_size
		b -= self.ipusironn*db / batch_size
		return W, b


class Decay:
	def __init__(self, ipusironn=0.42, lam=0.001):
		self.ipusironn = ipusironn
		self.lam = lam

	def __call__(self, W, b, dW, db, batch_size):
		W -= self.ipusironn * ((dW/batch_size) + self.lam * W)
		b -= self.ipusironn * ((db/batch_size) + self.lam * b)
		return W, b