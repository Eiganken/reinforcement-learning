import numpy as np

from collections import deque


MEMORY_SIZE = 10000
REPLAY_SIZE = 32


class Memory:
	"""
	経験を格納するクラス.

	Attributes
	----------
	memory : deque
		経験を格納する.
	replay_size : int
		ミニバッチサイズ.
	"""
	def __init__(self, memory_size=MEMORY_SIZE, replay_size=REPLAY_SIZE):
		self.memory = deque(maxlen=memory_size)
		self.replay_size = replay_size

	def add(self, experience):
		"""
		メモリーに経験を追加する.

		Parameters
		----------
		experience : tuple
			状態や行動などの対
		"""
		self.memory.append(experience)

	def get_sample(self):
		"""
		メモリーからサンプルを取り出す.
		"""
		idx = np.random.choice(np.arange(len(self.memory)),
							size=self.replay_size, replace=False)
		experiences = [self.memory[i] for i in idx]

		return experiences

	def is_able_fit(self):
		mask = len(self.memory) >= self.replay_size
		return mask