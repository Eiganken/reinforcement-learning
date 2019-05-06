import numpy as np

from .function import *
from .optimizer import *


class Layer:
	"""
	レイヤーを構成するクラス.
	
	Attributes
	----------
	layer_number : int
		層の順番.
	output_dim : int
		出力層の数.
	W : ndarray
		重み行列.
	b : ndarray
		バイアスベクトル.
	input_z, u : ndarray
		入力ベクトルと行列演算後のベクトル.
	dW, db : ndarray
		微分値(勾配)のベクトル.
	function : function
		活性化関数のクラス.
	optimizer : optimizer
		勾配法のクラス.
	"""
	def __init__(self, nords, activation):
		"""
		Parameters
		----------
		nords : int
			出力次元数.
		activation : str
			活性化関数の名前.
		"""
		self.layer_number = None
		self.output_dim = nords
		self.W, self.b = None, None
		self.input_z, self.u = None, None
		self.dW, self.db = None, None
		self.function = eval(activation)()
		self.optimizer = None

	def __call__(self, layer_number, input_dim, optimizer):
		"""
		初期化するためのコールバック関数.

		layer_number : int
			層の番号.
		input_dim : int
			入力次元数.
		optimizer : str
			降下法の名前.
		"""
		self.layer_number = layer_number
		self.W = np.random.randn(input_dim, self.output_dim)
		self.b = np.random.randn(self.output_dim)
		self.dW = 0
		self.db = 0
		self.optimizer = eval(optimizer)()

	def forward(self, z):
		"""
		順伝播方向に計算する.

		Parameters
		----------
		z : ndarray
			入力ベクトル.
		"""
		# 入力を保存
		self.input_z = z.reshape((-1,1))
		# 線形変換
		self.u = z@self.W + self.b
		# 活性化関数による出力を計算
		z = self.function.f(self.u)

		return z

	def backward(self, delta):
		"""
		逆伝播方向に計算する.勾配を求める.

		delta : ndarray
			逆伝播のデルタ.
		"""
		# 活性化関数の逆伝播
		delta = self.function.df(self.u) * delta
		# 勾配を計算
		self.dW += self.input_z @ delta.reshape((1,-1))
		self.db += delta
		# 出力を計算
		delta = self.W @ delta.T

		return delta

	def update(self, batch_size=32):
		"""
		勾配でパラメーターを更新する.

		Parameters
		----------
		batch_size : int
			ミニバッチのサイズ.
		"""
		# 勾配を正規化
		self.dW = self.dW / batch_size
		self.db = self.db / batch_size
		# 勾配法によりパラメーターを更新
		self.W, self.b = self.optimizer(self.W, self.b, self.dW, self.db)
		# 勾配を初期化
		self.dW, self.db = 0, 0

	def copy_parameter(self, original_layer):
		self.W, self.b = original_layer.W.copy(), original_layer.b.copy()


class OutputLayer(Layer):
	"""
	出力層
	"""
	def __init__(self, nords, activation):
		super().__init__(nords, activation)

	def backward(self, delta):
		"""
		逆伝播方向に計算する.勾配を求める.

		delta : ndarray
			逆伝播のデルタ. 誤差関数から出力を得ている.
		"""

		# 勾配を計算
		self.dW += np.dot(self.input_z, delta.reshape((1,-1)))
		self.db += delta
		# 出力を計算
		delta = self.W @ delta.T

		return delta