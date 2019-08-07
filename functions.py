import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vectors = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()
	mm_scaler = MinMaxScaler()
	for line in lines[1:]:
		# Date,Open,High,Low,Close,Adj Close,Volume
		split_data = line.split(",")
		_open = float(split_data[1])
		_high = float(split_data[2])
		_low = float(split_data[3])
		_close = float(split_data[4])
		_volume = float(split_data[6])
		vec = np.array([_open, _high, _low, _close, _volume])
		vectors.append(vec)

	vectors = mm_scaler.fit_transform(vectors)
	print("feature range of the data: {}".format(mm_scaler.feature_range))
	return vectors.tolist()


# returns the sigmoid
def sigmoid(gamma):
	if gamma < 0:
		return 1 - 1 / (1 + math.exp(gamma))
	else:
		return 1 / (1 + math.exp(-gamma))
	# return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		for j in range(5):
			# _tmp = sigmoid(block[i + 1][j] - block[i][j])
			_tmp = block[i + 1][j] - block[i][j]
			res.append(_tmp)

	block_data = np.array([res])
	return block_data
