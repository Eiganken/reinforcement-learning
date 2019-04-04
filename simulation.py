class Simulation:
	def __init__(self, env, agent):
		self.env = env
		self.agent = agent

	def train(self, eps_time, step_time, forever=False, show=False, info=False):
		if forever is True:
			episode = 0
			step_count = 0
			while step_count < 5:
				episode += 1
				if episode % 100 == 0:
					print("===== Episode {:>4} =====".format(episode))
				step = self._episode(step_time, episode, show, info, forever)
				if step > 195:
					step_count += 1
				else:
					step_count = 0
		else:
			for episode in range(1, eps_time+1):
				if episode % int(eps_time/10) == 0:
					print("===== Episode {:>4} =====".format(episode))
				self._episode(step_time, episode, show, info)

	def _episode(self, step_time, episode, show, info, forever=False):
		self.env.reset()
		state, _, _, _ = self.env.step(self.env.action_space.sample())
		self.agent.last_state = state

		sum_of_loss = 0

		for step in range(1, step_time+1):
			if show:
				self.env.render()

			action = self.agent.get_action(state, episode)

			state, _, done, _ = self.env.step(action)

			if done:
				if step < 195:
					reward = -1
				else:
					reward = 1
			else:
				reward = 0

			loss = self.agent.learn(state, reward, done)

			if loss is not None:
				sum_of_loss += loss

			if done:
				break

		if info:
			print("episode {:>3} || step time is : {:>3}  || loss mean is : {}"\
				  .format(episode, step, sum_of_loss/step))

		if forever:
			return step

	def test(self, eps_time, forever=False):
		if forever:
			while True:
				state = self.env.reset()
				done = False
				while not(done):
					self.env.render()
					action = self.agent.get_action(state, 10000)
					state, _, done, _ = self.env.step(action)
		else:
			for _ in range(eps_time):
				state = self.env.reset()
				done = False
				while not(done):
					self.env.render()
					action = self.agent.get_action(state, 10000)
					state, _, done, _ = self.env.step(action)