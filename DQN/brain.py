import numpy as np

from network.network import NeuralNetwork
from network.layer import Layer, OutputLayer

GAMMA = 0.99
TARGET_UPDATE_TIME = 5

class Brain:
	"""
	Q値の管理や学習を行う.
	"""
	def __init__(self, state_size, action_size):
		"""
		"""
		self.state_size = state_size
		self.action_size = action_size
		# action network の構築
		self.action_network = NeuralNetwork(state_size, "Huberloss", "Adam")
		self.action_network.add(Layer(16, activation="ReLU"))
		self.action_network.add(Layer(10, activation="ReLU"))
		self.action_network.add(OutputLayer(action_size, activation="Linear"))
		# target nework の構築
		self.target_network = self.action_network.copy_network()
		self.target_network.copy_parameters(self.action_network)

	def get_q_values(self, state):
		"""
		状態行動価値ベクトルを取得する.

		Parameters
		----------
		state : array
			状態ベクトル

		Returns
		-------
		q_values : ndarray
			状態行動価値ベクトル
		"""
		q_values = self.action_network.predict(state)

		return q_values

	def replay(self, episode, experiences, replay_size):
		"""
		経験再生を行う. 学習を行う.

		Parameters
		----------
		experiences : list
			経験リスト.
		replay_size : int
			ミニバッチのサイズ.
		"""
		# 学習データを取得
		x_train, t_train = self._get_train_data(experiences, replay_size)
		# 行動ネットワークの学習
		self.action_network.fit(x_train, t_train, epoch_time=1, batch_size=replay_size)
		# 教師ネットワークの更新
		if episode % TARGET_UPDATE_TIME == 0:
			self._update_network()

	def _get_train_data(self, experiences, replay_size):
		"""
		経験再生のための学習データを作成する.

		Parameters
		----------
		experiences : list
			経験リスト.
		replay_size : int
			ミニバッチのサイズ.

		Returns
		-------
		x_train : ndarray
			入力データ.
		t_train : ndarray
			教師データ.
		"""
		# 空の配列(データセット)の準備
		x_train = np.zeros((replay_size, self.state_size))
		t_train = np.zeros((replay_size, self.action_size))
		# データセットを作成する
		for i, (state, action, reward, next_state, done) in enumerate(experiences):
			# 終端状態とそうでない場合によって教師データの値を変える
			if done:
				target = reward
			else:
				# 行動ネットワークから次状態の最大価値となる行動を取得
				action_q_values = self.action_network.predict(next_state)
				next_action = np.argmax(action_q_values)
				# 教師ネットワークから次状態の価値を取得
				target_q_values = self.target_network.predict(next_state)
				# TD誤差(教師データ)を計算
				target = reward + GAMMA * target_q_values[next_action]
			# 教師データを保存する
			t_train[i] = self.action_network.predict(state)
			t_train[i][action] = target
			# 入力データを保存する
			x_train[i] = state
		
		return x_train, t_train

	def _update_network(self):
		"""
		教師ネットワークを更新する.
		"""
		self.target_network.copy_parameters(self.action_network)