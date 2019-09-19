""" Training thread for A3C
"""
import logging
import math

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


def train_custom_network(agent, input_data, scaler, thread_name, window_size, n_max, buy_amount, tqdm):
    """

    :param input_data:
    :param scaler:
    :param agent:
    :param batch_size:
    :param window_size:
    :param n_max:
    :param buy_amount:
    :param tqdm:
    :return:
    """
    thread_name = str(thread_name)
    batch_data = input_data
    logger = logging.getLogger('train_application_{}'.format(thread_name))
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs/thread{}.log'.format(thread_name))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    global episode
    budget = 1000
    t = 11
    while episode < n_max:
        data = input_data
        total_sample = len(data) - 1
        # print("\nEpisode " + str(episode) + "/" + str(n_max))
        order = {
            'price': 0,
            'raw_price': 0,
            'action': 0,
            'state': None,
            'next_state': None,
            'trading': False
        }
        state = getState(data, t, window_size + 1, 0, to_categorical(0, agent.act_dim))
        actions, states, rewards = [], [], []
        hold_time, cumul_reward, done = 0, 0, False
        # logger.info("Thread: {} | Totals sample: {}".format(thread_name, total_sample))
        while not done and episode < n_max and t < 65000:
            action = agent.policy_action(np.expand_dims(state, axis=0))
            # logger.info("Thread: {} | Predict Action: {}".format(thread_name, action))
            actions.append(to_categorical(action, agent.act_dim))
            states.append(state)
            current_stock_price = scaler.inverse_transform([data[t]])[0][3]
            order_price = order['price']
            next_state = getState(data, t + 1, window_size + 1, order_price, to_categorical(action, agent.act_dim))
            done = False
            reward = 1
            if action == 0:
                """Do nothing, reward of this action is zero"""
                pass

            if action == 1:  # Hold order
                if order['trading']:
                    if hold_time > 20:
                        done = True
                    else:
                        hold_time += 1
                        _reversed = 1
                        if order['action'] == 2:  # sell order
                            _reversed = -1
                        profit = (current_stock_price - order['raw_price']) * buy_amount * _reversed
                        state = next_state
                        reward = 1
                        msg = "Thread: " + thread_name + " | Hold order: " + formatPrice(
                            order['raw_price']) + " => " + formatPrice(
                            current_stock_price) + " | Profit: " + formatPrice(profit) + " Budget: " + str(budget)
                        logger.info(msg)
                else:
                    done = True
                    reward = 1

            elif action == 2:  # place buy order
                if order['trading']:
                    done = True
                    reward = 1
                else:
                    order = {
                        'price': data[t][3],
                        'raw_price': current_stock_price,
                        'action': action,
                        'state': state,
                        'next_state': next_state,
                        'trading': True
                    }
                    reward = 2
                    state = next_state
                    msg = "Thread: " + thread_name + " | Buy: " + formatPrice(current_stock_price)
                    logger.info(msg)

            elif action == 3:  # place sell order
                if order['trading']:
                    done = True
                    reward = 1
                else:
                    order = {
                        'price': data[t][3],
                        'raw_price': current_stock_price,
                        'action': action,
                        'state': state,
                        'next_state': next_state,
                        'trading': True
                    }
                    reward = 2
                    state = next_state
                    msg = "Thread: " + thread_name + " | Sell: " + formatPrice(current_stock_price)
                    logger.info(msg)
            elif action == 4:
                if not order['trading']:
                    done = True
                    reward = 1
                else:
                    _reversed = 1
                    if order['action'] == 2:  # sell order
                        _reversed = -1

                    profit = (current_stock_price - order['raw_price']) * buy_amount * _reversed
                    budget += profit
                    if profit > 0:
                        reward = 2
                        done = False
                    else:
                        reward = 1
                        done = True

                    order = {
                        'price': 0,
                        'raw_price': 0,
                        'action': 0,
                        'state': None,
                        'next_state': None,
                        'trading': False
                    }
                    state = getState(data, t, window_size + 1, 0, to_categorical(0, agent.act_dim))
                    msg = "Thread: " + thread_name + " | Close order: " + formatPrice(current_stock_price) + \
                          " | Profit: " + formatPrice(profit) + " Budget: " + str(budget)
                    logger.info(msg)

                    # reset hold time
                    hold_time = 0

            cumul_reward += reward
            rewards.append(reward)
            if done:
                # print("--------------------------------")
                # print("Budget: " + formatPrice(budget))
                # print("--------------------------------")
                logger.info('Thread: {} | Done | total_reward: {} | Budget: {}'.format(thread_name, cumul_reward, budget))
                order = {
                    'price': 0,
                    'raw_price': 0,
                    'action': 0,
                    'state': None,
                    'next_state': None,
                    'trading': False
                }
                budget = 1000
                lock.acquire()
                agent.train_models(states, actions, rewards, done)
                lock.release()
                actions, states, rewards = [], [], []

            t += 1
            # with lock:
            #     tqdm.set_description("\nScore: {}, Budget: {}".format(round(cumul_reward, 1), round(budget, 1)))
            #     tqdm.refresh()
        with lock:
            tqdm.update(1)
            if episode < n_max:
                episode += 1
