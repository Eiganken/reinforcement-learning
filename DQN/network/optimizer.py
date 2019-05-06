import numpy as np

class Gradient:
	"""
	確率的勾配降下法
	"""
	def __init__(self, ipusironn=0.4):
		self.ipusironn = ipusironn

	def __call__(self, W, b, dW, db):
		"""
		"""
		W -= self.ipusironn*dW
		b -= self.ipusironn*db
		return W, b


class Adam:
	"""
	Adam
	"""
	def __init__(self, alpha=0.00001, beta_first=0.9, beta_second=0.999, ipusironn=1.0e-8):
		self.alpha = alpha
		self.beta_first = beta_first
		self.beta_second = beta_second
		self.ipusironn = ipusironn

		self.W_m = 0
		self.W_v = 0
		self.b_m = 0
		self.b_v = 0
		self.t = 0.001

	def __call__(self, W, b, dW, db):
		# self.t += 

		self.W_m = self.beta_first*self.W_m + (1-self.beta_first)*dW
		self.W_v = self.beta_second*self.W_v + (1-self.beta_second)*dW**2
		self.b_m = self.beta_first*self.b_m + (1-self.beta_first)*db
		self.b_v = self.beta_second*self.b_v + (1-self.beta_second)*db**2

		W_m = self.W_m / (1-self.beta_first**self.t)
		W_v = self.W_v / (1-self.beta_second**self.t)
		b_m = self.b_m / (1-self.beta_first**self.t)
		b_v = self.b_v / (1-self.beta_second**self.t)

		W -= self.alpha*W_m / (W_v**0.5+self.ipusironn)
		b -= self.alpha*b_m / (b_v**0.5+self.ipusironn)

		return W, b