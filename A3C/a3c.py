import random
import threading
import time
from collections import deque

import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape

from functions import getStockDataVec
from .critic import Critic
from .actor import Actor
from .thread import train_custom_network
from utils.networks import conv_block


class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, gamma = 0.99, lr = 0.0001, is_atari=False):
        """ Initialization
        """
        # Environment and A3C parameters
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.99
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.epsilon = 0.5
        self.lr = lr
        # Create actor and critic networks
        self.memory = deque(maxlen=1000)
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input(shape=self.env_dim)
        # If we have an image, apply convolutional layers
        # 1D Inputs
        x = Dense(64, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        model = Model(inp, x)
        return model

    def policy_action(self, s):
        """ Use the actor's network to predict the next action to take, using the policy
        """
        p = self.actor.predict(s).ravel()
        p /= p.sum()
        return np.random.choice(np.arange(self.act_dim), 1, p=p)[0]

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.act_dim)

        options = self.actor.predict(state)
        action = np.argmax(options[0])
        # print("predict actions: {}, data: {}".format(action, options))
        return action

    def discount(self, r, done, s):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """
        Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, states[-1])
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))

        discounted_rewards = self.discount(rewards, done, states[-1])

        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self, args):
        # Instantiate one environment per thread
        stock_name, window_size, episode_count = args.stock_name, args.window_size, args.episode_count
        state_size = window_size * 5 + 2
        action_dim = args.action_dim

        stock_data = getStockDataVec(stock_name)
        batch_size = 32
        buy_amount = 1

        # Create threads
        tqdm_e = tqdm(range(int(args.episode_count)), desc='Score', leave=True, unit=" episodes")

        # data, agent, batch_size, window_size, n_max, buy_amount, tqdm
        # number_data = len(stock_data) // args.n_threads  # 5 threads
        # stock_data = np.array(stock_data)
        # stock_data = np.split(stock_data, number_data)
        threads = [threading.Thread(
            target=train_custom_network,
            daemon=True,
            args=(self,
                  stock_data,
                  batch_size,
                  window_size,
                  episode_count,
                  buy_amount,
                  tqdm_e)) for _ in range(args.n_threads)]

        for t in threads:
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")
        return None

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
