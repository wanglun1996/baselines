"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'actions', 'episode_returns', 'rewards', 'obs'
Old format: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import logger


class Dataset(object):
    def __init__(self, inputs, labels, randomize):
        """
        Dataset object

        :param inputs: (np.ndarray) the input values
        :param labels: (np.ndarray) the target values
        :param randomize: (bool) if the dataset should be shuffled
        """
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = 0
        self.init_pointer()

    def init_pointer(self):
        """
        initialize the pointer and shuffle the dataset, if randomize the dataset
        """
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        """
        get the batch from the dataset

        :param batch_size: (int) the size of the batch from the dataset
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class MujocoDataset(object):
    def __init__(self, expert_path, train_fraction=0.7,
                 traj_limitation=-1, randomize=True, verbose=1):
        """
        Dataset for mujoco

        :param expert_path: (str) the path to trajectory data
        :param train_fraction: (float) the train val split (0 to 1)
        :param traj_limitation: (int) the dims to load (if -1, load all)
        :param randomize: (bool) if the dataset should be shuffled
        :param verbose: (int) Verbosity
        """
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        observations = traj_data['obs'][:traj_limitation]
        actions = traj_data['acs'][:traj_limitation]

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.observations = np.reshape(observations, [-1, np.prod(observations.shape[2:])])
        self.actions = np.reshape(actions, [-1, np.prod(actions.shape[2:])])

        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets) / len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        self.verbose = verbose

        # if len(self.actions) > 2:
        #     self.actions = np.squeeze(self.actions)

        assert len(self.observations) == len(self.actions)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.observations)
        self.randomize = randomize
        self.dataset = Dataset(self.observations, self.actions, self.randomize)
        # for behavior cloning
        self.train_set = Dataset(self.observations[:int(self.num_transition * train_fraction), :],
                                 self.actions[:int(self.num_transition * train_fraction), :],
                                 self.randomize)
        self.val_set = Dataset(self.observations[int(self.num_transition * train_fraction):, :],
                               self.actions[int(self.num_transition * train_fraction):, :],
                               self.randomize)
        if self.verbose >= 1:
            self.log_info()

    def log_info(self):
        """
        log the information of the dataset
        """
        logger.log("Total trajectories: {}".format(self.num_traj))
        logger.log("Total transitions: {}".format(self.num_transition))
        logger.log("Average returns: {}".format(self.avg_ret))
        logger.log("Std for returns: {}".format(self.std_ret))

    def get_next_batch(self, batch_size, split=None):
        """
        get the batch from the dataset

        :param batch_size: (int) the size of the batch from the dataset
        :param split: (str) the type of data split (can be None, 'train', 'val')
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        if split is None:
            return self.dataset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        """
        Show histogram plotting of the episode returns
        """
        plt.hist(self.rets)
        plt.show()


def test(expert_path, traj_limitation, plot):
    """
    test mujoco dataset object

    :param expert_path: (str) the path to trajectory data
    :param traj_limitation: (int) the dims to load (if -1, load all)
    :param plot: (bool) enable plotting
    """
    dset = MujocoDataset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--expert_path', type=str, default='expert_pendulum.npz')
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
