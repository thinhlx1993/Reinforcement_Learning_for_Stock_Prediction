from agent.agent import Agent
from functions import *
import sys

# if len(sys.argv) != 4:
# 	print("Usage: python train.py [stock] [window] [episodes]")
# 	exit()

stock_name, window_size, episode_count = "XAUUSD15", 10, 10000
state_size = window_size*5 + 2
agent = Agent(state_size)
data = getStockDataVec(stock_name)
total_sample = len(data) - 1
batch_size = 128
buy_amount = 1

for e in range(episode_count + 1):
    order = {
        'price': 0,
        'action': 0,
        'state': None,
        'next_state': None,
        'trading': False
    }
    state = getState(data, 0, window_size + 1, order)

    total_profit = 0
    budget = 1000
    equity = 1000
    margin = 0

    for t in range(total_sample):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1, order)
        reward = 0
        done = False
        current_stock_price = data[t][3]
        if action == 0:
            if order['trading']:
                _reversed = 1
                if order['action'] == 2:  # sell order
                    _reversed = -1
                profit = (current_stock_price - order['price']) * buy_amount * _reversed
                reward = 1 if profit > 0 else -1
                print("Hold order: " + formatPrice(order['price']) + " => " + formatPrice(current_stock_price) + " | Profit: " + formatPrice(profit))
            else:
                reward = -1

        elif action == 1:  # place order buy
            if order['trading']:
                reward = -5
            else:
                margin += agent.calculate_margin(current_stock_price, buy_amount)
                order = {
                    'price': current_stock_price,
                    'action': action,
                    'state': state,
                    'next_state': next_state,
                    'trading': True
                }
                reward = 1
                print("Buy: " + formatPrice(current_stock_price))
        elif action == 2:  # place order sell
            if order['trading']:
                reward = -5
            else:
                margin += agent.calculate_margin(current_stock_price, buy_amount)
                order = {
                    'price': current_stock_price,
                    'action': action,
                    'state': state,
                    'next_state': next_state,
                    'trading': True
                }
                reward = 1
                print("Sell: " + formatPrice(current_stock_price))

        elif action == 3:  # close order
            if not order['trading']:
                reward = -5
            else:
                _reversed = 1
                if order['action'] == 2:  # sell order
                    _reversed = -1

                profit = (current_stock_price - order['price']) * buy_amount * _reversed
                reward = 1 if profit > 0 else -1

                #if reward < 0:
                #    agent.memory.append((order['state'], order['action'], reward, order['next_state'], True))

                budget += profit
                order = {
                    'price': 0,
                    'action': 0,
                    'state': None,
                    'next_state': None,
                    'trading': False
                }
                print("Close order: " + formatPrice(current_stock_price) + " | Profit: " + formatPrice(profit))

                done = True if (profit < 0) else False
                
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Episode " + str(e) + "/" + str(episode_count) + " Timestep: " + str(t))
            print("Budget: " + formatPrice(budget))
            print("--------------------------------")
            order = {
                'price': 0,
                'action': 0,
                'state': None,
                'next_state': None,
                'trading': False
            }
            #budget = 1000
            #equity = 1000
            #margin = 0

        if len(agent.memory) > batch_size and t % 10 == 0:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.actor.save("models/model_ep" + str(e))
