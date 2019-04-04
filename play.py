import gym
from agent import Agent

from simulation import Simulation

def simulate():
	env = gym.make('CartPole-v0')
	agent = Agent(4, 2)

	simulation = Simulation(env, agent)

	simulation.train(300, 200, show=False, info=True, forever=True)

	simulation.test(100, forever=True)

	env.close()

if __name__ == "__main__":
	simulate()