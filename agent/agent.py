import keras
from keras import Input
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Conv1D
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.windows = 10

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.actor_lr = 0.0001
        self.discount_factor = .9

        self.actor = load_model("models/" + model_name) if is_eval else self._actor()
        self.critic = self._critic()

    def _actor(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.state_size))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size))

        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def _critic(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.state_size))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def act(self, state):
            if not self.is_eval and random.random() <= self.epsilon:
                return random.randrange(self.action_size)

            options = self.actor.predict(state)
            action = np.argmax(options[0])
            print("predict actions: {}, data: {}".format(action, options))
            return action

    def expReplay(self, batch_size):
        mini_batch = []
        len_of_memory = len(self.memory)
        for i in range(len_of_memory - batch_size + 1, len_of_memory):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            advantages = np.zeros((1, self.action_size))
            target = self.critic.predict(state)[0]
            next_target = self.critic.predict(next_state)[0]

            advantages[0][action] = reward - target
            target[0] = reward

            if not done:
                # target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                advantages[0][action] = reward + (self.gamma * next_target) - target
                target[0] = reward + self.discount_factor * next_target

            # target_f = self.model.predict(state)
            # target_f[0][action] = target
            self.actor.fit(state, advantages, epochs=1, verbose=0)
            self.critic.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
