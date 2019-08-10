import numpy as np


class TradingEnv(object):
    def __init__(self, input_dim, action_dim, consecutive_frames, stock_name):
        self.inp_dim = input_dim
        self.act_dim = action_dim
        self.epsilon = 1
        self.consecutive_frames = consecutive_frames
        self.data = self.get_stock_data(stock_name)
        self.budget = 1000
        self.order = {
            'price': 0,
            'action': 0
        }

    def step(self, t, a):
        # returns an an n-day state representation ending at time t
        d = t - self.consecutive_frames + 1
        block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1]  # pad with t0
        states = []
        for i in range(self.consecutive_frames):
            res = []
            for j in range(self.inp_dim[0]):
                res.append(block[i][j])
            states.append(res)

        reward, done = 1, False
        if a == 1 or a == 2:
            if self.order['price'] != 0:
                reward, done = -1, True
            else:
                # place order
                self.order = {
                    'price': self.data[t][3],
                    'action': a
                }
                reward, done = 1, False

        if a == 3 or a == 0:
            # close order
            reward, done = self.cal_reward(t, a)

        return np.array(states), reward, done, {}

    def cal_reward(self, timestep, action):
        if action == 0:
            current_step = self.data[timestep]
            prev_step = self.data[timestep-1]
            reward = current_step[3] - prev_step[3]
            if self.order['action'] == 2:
                reward = (reward*-1)

        if action == 3:
            current_price = self.data[timestep][3]
            diff = self.order['price'] - current_price
            if self.order['action'] == 2:
                diff = diff * -1

            self.budget += diff
            self.order = {
                'price': 0,
                'action': 0
            }

            reward = diff

        done = True if reward < 0 else False

        return reward, done

    def reshape(self, x):
        if len(x.shape) < 4 and len(self.inp_dim) > 2:
            return np.expand_dims(x, axis=0)
        elif len(x.shape) < 2:
            return np.expand_dims(x, axis=0)
        else:
            return x

    def reset(self, t):
        # returns an an n-day state representation ending at time t
        d = t - self.consecutive_frames + 1
        block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1]  # pad with t0
        states = []
        for i in range(self.consecutive_frames):
            res = []
            for j in range(self.inp_dim[0]):
                res.append(block[i][j])
            states.append(res)

        return np.array(states)

    @staticmethod
    def format_price(n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

    @staticmethod
    def get_stock_data(key):
        """
        # returns the vector containing stock data from a fixed file
        :param key:
        :return:
        """
        vectors = []
        lines = open("data/" + key + ".csv", "r").read().splitlines()
        for line in lines[1:]:
            # Date,Open,High,Low,Close,Adj Close,Volume
            split_data = line.split(",")
            _open = float(split_data[2])
            _high = float(split_data[3])
            _low = float(split_data[4])
            _close = float(split_data[5])
            _volume = float(split_data[6])
            vec = np.array([_open, _high, _low, _close])
            vectors.append(vec)
        return vectors
