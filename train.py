import argparse
from A3C.a3c import A3C
import sys
from keras.backend.tensorflow_backend import set_session, get_session


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--window_size', type=int, default=10, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--stock_name', type=str, default='XAUUSD15', help="Name of stock")
    parser.add_argument('--n_threads', type=int, default=16, help="Number of threads (A3C)")
    parser.add_argument('--episode_count', type=int, default=500000, help="Number of episode")
    parser.add_argument('--action_dim', type=int, default=4, help="action dim")
    parser.add_argument('--env_dim', type=tuple, default=(52,), help="env_dim dim")
    return parser.parse_args(args)


if __name__ == '__main__':
    set_session(get_session())
    args = sys.argv[1:]
    args = parse_args(args)
    network = A3C(act_dim=args.action_dim, env_dim=args.env_dim)
    network.train(args=args)
    network.save_weights('models/a3c')
