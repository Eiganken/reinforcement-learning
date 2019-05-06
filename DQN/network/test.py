import numpy as np
import pandas as pd

from network import NeuralNetwork
from layer import Layer, OutputLayer

EPOCH_TIME = 2000
BATCH_SIZE = 64

def test():
	data = pd.read_csv("train.csv").values
	x = data[:,1:]
	t = np.identity(10, dtype=np.uint8)[data[:,0]]

	x = (x - x.min()) / x.max()
	x = (x - x.mean()) / x.std()

	validate = int(x.shape[0]*0.75)

	x_train, t_train = x[:validate], t[:validate]
	x_test, t_test = x[validate:], t[validate:]

	network = NeuralNetwork(784, error="R2error", optimizer="Gradient")
	network.add(Layer(100, activation="Sigmoid"))
	network.add(Layer(50, activation="Sigmoid"))
	network.add(OutputLayer(10, activation="Softmax"))

	network.fit(x_train, t_train, epoch_time=EPOCH_TIME, batch_size=BATCH_SIZE)

	network.print_accurate(x_test, t_test)

if __name__ == "__main__":
	test()