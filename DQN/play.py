import gym

from agent import Agent
from simulation import Simulation


def test():
	environment = gym.make('CartPole-v0')
	state_size = environment.observation_space.shape[0]
	action_size = environment.action_space.n
	agent = Agent(state_size=state_size, action_size=action_size)

	simulation = Simulation(environment, agent)

	simulation.train(500, 200)
	simulation.plot_step_times()

if __name__=="__main__":
	test()