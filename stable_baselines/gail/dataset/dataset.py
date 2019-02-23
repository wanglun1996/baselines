"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'actions', 'episode_returns', 'rewards', 'obs',
'episode_starts'
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
        self.num_samples = len(inputs)
        self.pointer = 0
        self.init_pointer()

    def init_pointer(self):
        """
        initialize the pointer and shuffle the dataset, if randomize the dataset
        """
        self.pointer = 0
        # Shuffle the dataset
        if self.randomize:
            indices = np.random.permutation(self.num_samples)
            self.inputs = self.inputs[indices, :]
            self.labels = self.labels[indices, :]

    def get_next_batch(self, batch_size):
        """
        get the batch from the dataset

        :param batch_size: (int) the size of the batch from the dataset
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_samples:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class ExpertDataset(object):
    def __init__(self, expert_path, train_fraction=0.7,
                 traj_limitation=-1, randomize=True, verbose=1):
        """
        Dataset for using behavior cloning or GAIL.
        NOTE: Images input are not supported properly for now.

        :param expert_path: (str) the path to trajectory data
        :param train_fraction: (float) the train val split (0 to 1)
        :param traj_limitation: (int) the dims to load (if -1, load all)
        :param randomize: (bool) if the dataset should be shuffled
        :param verbose: (int) Verbosity
        """
        # TODO: properly support images as input
        # (but too much memory usage for now, need a dataloader)
        # TODO: support discrete actions (convert to one hot encoding)
        traj_data = np.load(expert_path)

        if verbose > 0:
            for key, val in traj_data.items():
                print(key, val.shape)

        # Array of bool where episode_starts[i] = True for each new episode
        episode_starts = traj_data['episode_starts']

        traj_limit_idx = len(traj_data['obs'])

        if traj_limitation > 0:
            n_episodes = 0
            # Retrieve the index corresponding
            # to the traj_limitation trajectory
            for idx, episode_start in enumerate(episode_starts):
                n_episodes += int(episode_start)
                if n_episodes == (traj_limitation + 1):
                    traj_limit_idx = idx - 1

        observations = traj_data['obs'][:traj_limit_idx]
        actions = traj_data['actions'][:traj_limit_idx]

        # obs, actions: shape (N * L, ) + S
        # where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # S = (1, ) for discrete space
        # Flatten to (N * L, prod(S))
        if len(actions.shape) > 2:
            observations = np.reshape(observations, [-1, np.prod(observations.shape[1:])])
            actions = np.reshape(actions, [-1, np.prod(actions.shape[1:])])

        self.observations = observations
        self.actions = actions

        self.returns = traj_data['episode_returns'][:traj_limit_idx]
        self.avg_ret = sum(self.returns) / len(self.returns)
        self.std_ret = np.std(np.array(self.returns))
        self.verbose = verbose

        assert len(self.observations) == len(self.actions)
        self.num_traj = min(traj_limitation, np.sum(episode_starts))
        self.num_transition = len(self.observations)
        self.randomize = randomize
        self.dataset = Dataset(self.observations, self.actions, self.randomize)
        # For behavior cloning
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
        Log the information of the dataset
        """
        logger.log("Total trajectories: {}".format(self.num_traj))
        logger.log("Total transitions: {}".format(self.num_transition))
        logger.log("Average returns: {}".format(self.avg_ret))
        logger.log("Std for returns: {}".format(self.std_ret))

    def get_next_batch(self, batch_size, split=None):
        """
        Get the batch from the dataset

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
        plt.hist(self.returns)
        plt.show()


def test(expert_path, traj_limitation, plot):
    """
    test mujoco dataset object

    :param expert_path: (str) the path to trajectory data
    :param traj_limitation: (int) the dims to load (if -1, load all)
    :param plot: (bool) enable plotting
    """
    dset = ExpertDataset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--expert_path', type=str, default='expert_pendulum.npz')
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
