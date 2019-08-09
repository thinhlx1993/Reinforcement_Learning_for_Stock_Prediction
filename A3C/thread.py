""" Training thread for A3C
"""

import numpy as np
from threading import Thread, Lock
from keras.utils import to_categorical

from functions import getState, formatPrice
from utils.networks import tfSummary
import random
episode = 0
lock = Lock()


def training_thread(agent, Nmax, env, action_dim, f, summary_writer, tqdm, render):
    """ Build threads to run shared computation across
    """

    global episode
    while episode < Nmax:

        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards = [], [], []
        while not done and episode < Nmax:
            if render:
                with lock: env.render()
            # Actor picks an action (following the policy)
            a = agent.policy_action(np.expand_dims(old_state, axis=0))
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Memorize (s, a, r) for training
            actions.append(to_categorical(a, action_dim))
            rewards.append(r)
            states.append(old_state)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1
            # Asynchronous training
            if(time%f==0 or done):
                lock.acquire()
                agent.train_models(states, actions, rewards, done)
                lock.release()
                actions, states, rewards = [], [], []

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=episode)
        summary_writer.flush()
        # Update episode count
        with lock:
            tqdm.set_description("Score: " + str(cumul_reward))
            tqdm.update(1)
            if(episode < Nmax):
                episode += 1


def train_custom_network(agent, input_data, batch_size, window_size, n_max, buy_amount, tqdm):
    """

    :param input_data:
    :param agent:
    :param batch_size:
    :param window_size:
    :param n_max:
    :param buy_amount:
    :param tqdm:
    :return:
    """
    batch_data = np.split(np.array(input_data), 5000)
    global episode
    while episode < n_max:
        data = random.choice(batch_data).tolist()
        total_sample = len(data) - 1
        # print("\nEpisode " + str(episode) + "/" + str(n_max))
        order = {
            'price': 0,
            'action': 0,
            'state': None,
            'next_state': None,
            'trading': False
        }
        cumul_reward = 0
        state = getState(data, 0, window_size + 1, order)
        budget = 1000
        actions, states, rewards = [], [], []
        for t in range(total_sample):
            action = agent.policy_action(np.expand_dims(np.array(state), axis=0))
            actions.append(to_categorical(action, 4))
            states.append(state)
            next_state = getState(data, t + 1, window_size + 1, order)
            reward = 0

            current_stock_price = data[t][3]
            if action == 0:
                if order['trading']:
                    _reversed = 1
                    if order['action'] == 2:  # sell order
                        _reversed = -1
                    profit = (current_stock_price - order['price']) * buy_amount * _reversed
                    reward = profit
                    # print("Hold order: " + formatPrice(order['price']) + " => " + formatPrice(
                    #     current_stock_price) + " | Profit: " + formatPrice(profit))
                else:
                    reward = -1

            elif action == 1:  # place order buy
                if order['trading']:
                    reward = -5
                else:
                    order = {
                        'price': current_stock_price,
                        'action': action,
                        'state': state,
                        'next_state': next_state,
                        'trading': True
                    }
                    reward = 1
                    # print("Buy: " + formatPrice(current_stock_price))
            elif action == 2:  # place order sell
                if order['trading']:
                    reward = -5
                else:
                    order = {
                        'price': current_stock_price,
                        'action': action,
                        'state': state,
                        'next_state': next_state,
                        'trading': True
                    }
                    reward = 1
                    # print("Sell: " + formatPrice(current_stock_price))

            elif action == 3:  # close order
                if not order['trading']:
                    reward = -5
                else:
                    _reversed = 1
                    if order['action'] == 2:  # sell order
                        _reversed = -1

                    profit = (current_stock_price - order['price']) * buy_amount * _reversed
                    reward = profit

                    if reward < 0:
                        agent.train_models([order['state']], [order['action']], [reward], True)

                    budget += profit
                    order = {
                        'price': 0,
                        'action': 0,
                        'state': None,
                        'next_state': None,
                        'trading': False
                    }
                    # print("Close order: " + formatPrice(current_stock_price) + " | Profit: " + formatPrice(profit))

            done = True if (budget < 0) else False
            # agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            cumul_reward += reward
            rewards.append(reward)
            lock.acquire()
            agent.train_models(states, actions, rewards, done)
            lock.release()
            if done:
                # print("--------------------------------")
                # print("Budget: " + formatPrice(budget))
                # print("--------------------------------")
                # order = {
                #     'price': 0,
                #     'action': 0,
                #     'state': None,
                #     'next_state': None,
                #     'trading': False
                # }
                # budget = 1000
                # actions, states, rewards = [], [], []
                break

        with lock:
            tqdm.set_description("\nScore: {}, Budget: {}".format(round(cumul_reward, 1), round(budget, 1)))
            tqdm.update(1)
        if episode < n_max:
            episode += 1
