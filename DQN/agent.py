import numpy as np

from brain import Brain
from memory import Memory

class Agent:
	"""
	エージェントクラス

	Attributes
	----------
	brain : brain
	memory : memory
	replay_size : int
		経験再生時に取り出す経験データの数
	last_state : ndarray
	last_action : int
	episode : int
		動的に追加する. それぞれ状態,行動,エピソードの番号を保存
	"""
	def __init__(self, state_size, action_size, replay_size=32):
		"""
		state_size : int
			状態空間の次元数
		action_size : int
			行動空間の次元数
		"""
		self.brain = Brain(state_size, action_size)
		self.memory = Memory()
		self.replay_size = replay_size

	def get_action(self, state, episode, optimal=False):
		"""
		行動を決定する.

		Parameters
		----------
		state : list
			状態ベクトル.
		"""
		# 方策により行動する
		if np.random.rand() < 0.001 + 0.9 / (1.0+episode):
			action = np.random.randint(self.brain.action_size)
		else:
			# Q値を取得
			q_values = self.brain.get_q_values(state)
			action = np.argmax(q_values)
		# 状態と行動を保存する
		self.last_state = state
		self.last_action = action
		self.episode = episode

		return action

	def learn(self, reward, next_state, done):
		"""
		学習を行う.

		Parameters
		----------
		reward : たぶんint
			報酬
		state : array
			状態ベクトル
		done : bool
			終端状態かどうか
		"""
		# 経験をメモリーに保存する
		experience = (self.last_state, self.last_action, reward, next_state, done)
		self.memory.add(experience)
		# 経験再生
		if self.memory.is_able_fit():
			experiences = self.memory.get_sample()
			self.brain.replay(self.episode, experiences, self.replay_size)