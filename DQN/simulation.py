import sys
import numpy as np
import matplotlib.pyplot as plt

class Simulation:
	"""
	環境とエージェントから強化学習の環境を構築するクラス.

	Attibutes
	---------
	env : Environment
		環境オブジェクト. OpenAIのenvかUnityMLAgentsから取得したenvを用いる.
	agent : Agent
		エージェントオブジェクト.
	"""
	def __init__(self, env, agent):
		self.env = env
		self.agent = agent

		self.rewards = []

	def train(self, eps_time, max_step_num):
		for episode in range(1, eps_time+1):
			self._episode(episode, max_step_num)
			self._print_bar(episode, eps_time, 50)
		self.env.close()

	def _episode(self, episode, max_step_num):
		"""
		1エピソードを行う関数
		"""
		# 環境を初期化する
		observation = self.env.reset()
		# 初期行動をランダムにする
		action = np.random.randint(self.env.action_space.n)
		observation, _, done, _ = self.env.step(action)

		for step in range(max_step_num):
			# 環境を描画
			# self.env.render()
			# エージェントから行動を取得
			action = self.agent.get_action(observation, episode)
			# 環境に行動を渡す
			next_observation, reward, done, _ = self.env.step(action)
			# 報酬の設計
			if done:
				if step < max_step_num*0.95:
					reward = -1
				else:
					reward = 1
			else:
				reward = 0
			# 学習
			self.agent.learn(reward, next_observation, done)
			# observationを更新
			observation = next_observation
			# 終端状態なら終了する
			if done:
				self.rewards.append(reward)
				break

	def _print_bar(self, episode, episode_time, size):
		"""
		学習経過を可視化する.

		Parameters
		----------
		episode : int
			現在のエポック.
		episode_time : int
			最大エポック数.
		size : int
			ビン(ハイフン)の数.
		"""
		if episode == 1 or episode % (episode_time/size) == 0:
			# ビンの個数を計算
			mask = int(episode/(episode_time/size))
			# バーの文字列を保存
			self.bar_strings = "[" + "#"*mask + "-"*(size-mask) + "]"
		# 進捗率を計算
		persentage = 100 * episode//episode_time
		sys.stdout.write("\r"+self.bar_strings+" "+str(persentage)+"%")
		if persentage == 100:
			sys.stdout.write("\n")

	def plot_step_times(self):
		mean_rewards = [sum(self.rewards[i:i+10])/10 for i in range(0,len(self.rewards),10)]
		episodes = [i for i in range(0, len(self.rewards), 10)]
		plt.plot(episodes, mean_rewards)
		plt.show()