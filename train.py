from agent.agent import Agent
from functions import *
import sys

# if len(sys.argv) != 4:
# 	print("Usage: python train.py [stock] [window] [episodes]")
# 	exit()

stock_name, window_size, episode_count = "^GSPC", 10, 10000
state_size = window_size*5
agent = Agent(state_size)
data = getStockDataVec(stock_name)
total_sample = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(total_sample):
		action = agent.act(state)
		next_state = getState(data, t + 1, window_size + 1)
		reward = 1

		price = data[t][3]
		if action == 1: # buy
			if len(agent.inventory) > 5:
				reward = -10
			else:
				reward = 10
				agent.inventory.append(price)
			# print("Buy: " + formatPrice(price))

		elif action == 2 and len(agent.inventory) > 0:  # sell
			bought_price = agent.inventory.pop(0)
			reward = -100 if (price - bought_price) < 0 else 100
			total_profit += price - bought_price
			# print("Sell: " + formatPrice(price) + " | Profit: " + formatPrice(price - bought_price))

		done = True if reward < 0 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")
			total_profit = 0
			agent.inventory = []

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.actor.save("models/model_ep" + str(e))
