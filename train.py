import sys
import argparse
import tensorflow as tf
import pandas as pd

from A3C.a3c import A3C
from DDQN.ddqn import DDQN
from keras.backend.tensorflow_backend import set_session, get_session
from stable_baselines.common.vec_env import DummyVecEnv

from env import TradingEnv
from env.StockTradingEnv import StockTradingEnv


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--window_size', type=int, default=10, help="Stack frame")
    parser.add_argument('--stock_name', type=str, default='XAUUSD15', help="Name of stock")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    parser.add_argument('--episode_count', type=int, default=5000000, help="Number of episode")
    parser.add_argument('--action_dim', type=int, default=4, help="action dim")
    parser.add_argument('--state_dim', type=tuple, default=(10,), help="env_dim dim")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    parser.add_argument('--consecutive_frames', type=int, default=10,
                        help="Number of consecutive frames (action repeat)")
    parser.add_argument('--nb_episodes', type=int, default=5000000, help="Number of training episodes")
    return parser.parse_args(args)


def a3c(args):
    set_session(get_session())
    network = A3C(act_dim=args.action_dim, env_dim=args.state_dim, windows_size=args.window_size)
    network.train(args=args)
    network.save_weights('models/a3c')


def ddqn(args):
    summary_writer = tf.summary.FileWriter("DDQN/tensorboard_trading")
    env = TradingEnv(input_dim=args.state_dim, action_dim=args.action_dim,
                     consecutive_frames=args.consecutive_frames, stock_name=args.stock_name)
    algo = DDQN(args.action_dim, args.state_dim, args)
    algo.train(env, args, summary_writer)
    algo.save_weights('models/a3c')


if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)
    ddqn(args)
