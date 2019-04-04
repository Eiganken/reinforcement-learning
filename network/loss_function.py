import numpy as np

class MeanSquareLoss:
	def __init__(self):
		pass

	def f(self, y, d):
		loss = y - d
		return np.dot(loss, loss)/2
	
	def df(self, y, d):
		return y - d


class HuberLoss:
	def __init__(self, ipusironn=0.5):
		self.ipusironn = ipusironn
		self.mask_positive = None
		self.mask_negative = None

	def f(self, y, d):
		self.mask_positive = (y - d > self.ipusironn)
		self.mask_negative = (y - d < -self.ipusironn)
		mask = self.mask_positive | self.mask_negative
		L1 = np.square(y - d) * 0.5
		L2 = self.ipusironn * (np.abs(y-d) - self.ipusironn * 0.5)

		loss = np.where(mask, L2, L1)

		return np.sum(loss)

	def df(self, y, d):
		delta = y - d
		delta[self.mask_positive] = self.ipusironn
		delta[self.mask_negative] = -self.ipusironn
		return delta