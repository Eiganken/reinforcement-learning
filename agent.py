import numpy as np

from collections import deque

from network.model import NeuralNetwork
from network.layer import Layer
from network.activate_function import *
from network.loss_function import *
from network.optimizer import *

GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 32

class Agent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size

		self.main_network = Brain(state_size, action_size)
		self.target_network = Brain(state_size, action_size)
		self.main_network.copy(self.target_network)

		self.memory = Memory()

		self.last_state = None
		self.last_action = None

	def get_action(self, state, episode):
		if np.random.rand() < 0.001 + 0.9 / (1+episode):
			action = np.random.randint(self.action_size)
		else:
			q_values = self.main_network.get_q_values(state)
			action = np.argmax(q_values)
		
		self.last_state = state
		self.last_action = action

		return action

	def learn(self, state, reward, done):
		self.memory.add((self.last_state, self.last_action, reward, state, done))

		if not(self.memory.is_able_fit()):
			return
		
		x_train, t_train = self._get_train_set()

		loss = self.main_network.fit(x_train, t_train)

		self.main_network.copy(self.target_network)

		return loss

	def _get_train_set(self):
		x_train = np.zeros((BATCH_SIZE, self.state_size))
		t_train = np.zeros((BATCH_SIZE, self.action_size))

		mini_batch = self.memory.sample()

		for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
			target = reward

			if not(done):
				main_q_values = self.main_network.get_q_values(next_state)
				next_action = np.argmax(main_q_values)
				target_q_values = self.target_network.get_q_values(next_state)
				target = reward + GAMMA * target_q_values[next_action]

			t_train[i] = self.main_network.get_q_values(state)
			t_train[i][action] = target

			x_train[i] = state

		return x_train, t_train

class Brain:
	def __init__(self, state_size, action_size):

		L1 = Layer((state_size, 25), ReLU(), Adam(alpha=0.001))
		L2 = Layer((25, 20), ReLU(), Adam(alpha=0.001))
		OL = Layer((20, action_size), Linear(), Adam(alpha=0.001), loss=HuberLoss())

		self.network = NeuralNetwork(L1, L2, OL, batch_size=BATCH_SIZE)

	def get_q_values(self, state):
		return self.network.predict(state)

	def fit(self, x_train, t_train):
		mean_loss = self.network.fit(x_train, t_train)
		return mean_loss

	def copy(self, brain):
		brain.network = self.network.copy()

class Memory:
	def __init__(self):
		self.buffer = deque(maxlen=MEMORY_SIZE)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self):
		idx = np.random.choice(np.arange(len(self.buffer)), size=BATCH_SIZE, replace=False)
		return [self.buffer[i] for i in idx]

	def is_able_fit(self):
		return len(self.buffer) >= BATCH_SIZE