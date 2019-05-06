import sys
import numpy as np

from .function import *
from .layer import *

class NeuralNetwork:
	"""
	ニューラルネットワークを構成するクラス

	Attributes
	----------
	layers : list
		レイヤーを格納するリスト.
	input_dim : int
		入力次元数.
	loss_function : function
		誤差関数.
	optimizer : str
		勾配法の名前.
	bar_strings : str
		barの文字列.
	"""
	def __init__(self, input_dim, error, optimizer):
		"""
		Parameters
		----------
		input_dim : int
			入力の次元数.
		error : str
			誤差関数の名前.
		optimizer : str
			勾配法の名前.
		"""
		self.layers = []
		self.input_dim = input_dim
		self.loss_function = eval(error)()
		self.optimizer = optimizer

		self.bar_strings = None

	def add(self, layer):
		"""
		ネットワークに層を追加する.

		Parameters
		----------
		layer : layer
			レイヤークラス.
		"""
		# layerの情報
		layer_number = len(self.layers) + 1
		if len(self.layers) > 0:
			input_dim = self.layers[-1].output_dim
		else:
			input_dim = self.input_dim
		optimizer = self.optimizer
		# layerの初期化
		layer(layer_number, input_dim, optimizer)
		# ネットワークに追加
		self.layers.append(layer)

	def predict(self, x):
		"""
		順伝播を行う

		Parameters
		----------
		x : ndarray
			入力ベクトル.
		"""
		# 順伝播
		z = x
		for layer in self.layers:
			z = layer.forward(z)

		return z

	def evaluate(self, y, d):
		"""
		逆伝播を行う.

		Parameters
		----------
		y : ndarray
			予測ベクトル.
		d : ndarray
			教師ベクトル.
		"""
		# 誤差関数の逆伝播 -> 損失和を
		delta = self.loss_function.f(y,d) * self.loss_function.df(y,d)
		# 逆伝播
		for layer in self.layers[::-1]:
			delta = layer.backward(delta)
	
	def fit(self, x_train, d_train, epoch_time, batch_size):
		"""
		学習を行う
		
		Parameters
		----------
		x_train : ndarray
			訓練データの入力ベクトルを格納している2次元配列.
		d_train : ndarray
			訓練データの教師ベクトル.
		epoch_time : int
			最大エポック数.
		batch_size : int
			ミニバッチ数.
		"""
		# エポック数だけミニバッチ学習する
		for epoch in range(1, epoch_time+1):
			# バーを出力
			# self._print_bar(epoch, epoch_time, 50)
			# バッチデータを取得
			batch_mask = np.random.randint(x_train.shape[0], size=batch_size)
			x_batch, d_batch = x_train[batch_mask], d_train[batch_mask]
			# ミニバッチ学習
			for x, d in zip(x_batch, d_batch):
				# 予測を計算
				y = self.predict(x)
				# 勾配を計算
				self.evaluate(y, d)
			# パラメータを更新
			for layer in self.layers:
				layer.update(batch_size)

	def print_accurate(self, x_test, d_test):
		"""
		データから精度を算出する.

		Parameters
		----------
		x_test : ndarray
			入力ベクトルを格納している2次元配列.
		d_test : ndarray
			教師ベクトル.
		"""
		# 予測
		y_predict = [self.predict(x) for x in x_test]
		# 精度を算出
		accurate = [np.argmax(y) == np.argmax(d) for y, d in zip(y_predict, d_test)]
		accurate = np.mean(accurate)
		print("accurate : {}".format(accurate))

	def _print_bar(self, epoch, epoch_time, size):
		"""
		学習経過を可視化する.

		Parameters
		----------
		epoch : int
			現在のエポック.
		epoch_time : int
			最大エポック数.
		size : int
			ビン(ハイフン)の数.
		"""
		if epoch == 1 or epoch % (epoch_time/size) == 0:
			# ビンの個数を計算
			mask = int(epoch/(epoch_time/size))
			# バーの文字列を保存
			self.bar_strings = "[" + "#"*mask + "-"*(size-mask) + "]"
		# 進捗率を計算
		persentage = 100 * epoch//epoch_time
		sys.stdout.write("\r"+self.bar_strings+" "+str(persentage)+"%")
		if persentage == 100:
			sys.stdout.write("\n")

	def copy_network(self):
		"""
		ネットワークのコピーを返す.]

		Returns
		-------
		network : NeuralNetwork
			ネットワークのコピーオブジェクト.
		"""
		# 情報を格納
		input_dim = self.input_dim
		error = self.loss_function.__class__.__name__
		optimizer = self.optimizer
		# ネットワークのコピーを作成
		network = NeuralNetwork(input_dim, error, optimizer)
		# レイヤーのコピーを追加
		for i, layer in enumerate(self.layers):
			# レイヤーのコンストラクタ
			nords = layer.output_dim
			activation = layer.function.__class__.__name__
			# レイヤーを追加
			if i < len(self.layers)-1:
				network.add(Layer(nords, activation=activation))
			else:
				network.add(OutputLayer(nords, activation=activation))

		return network

	def copy_parameters(self, original_network):
		"""
		パラメータをコピーする

		Parameters
		----------
		original_network : NeuralNetwork
			コピー元のネットワークオブジェクト
		"""
		for layer, original_layer in zip(self.layers, original_network.layers):
			layer.copy_parameter(original_layer)