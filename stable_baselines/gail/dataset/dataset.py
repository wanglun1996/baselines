"""
Data structure of the expert .npz:
the data is save in python dictionary format with keys: 'actions', 'episode_returns', 'rewards', 'obs',
'episode_starts'
In case of images, 'obs' contains the relative path to the images.
"""
import queue
import time
from multiprocessing import Queue, Process

import cv2
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from stable_baselines import logger


class ExpertDataset(object):
    """
    Dataset for using behavior cloning or GAIL.

    :param expert_path: (str) the path to trajectory data (.npz file)
    :param train_fraction: (float) the train validation split (0 to 1)
        for pre-training using behavior cloning (BC)
    :param batch_size: (int) the minibatch size for behavior cloning
    :param traj_limitation: (int) the number of trajectory to use (if -1, load all)
    :param randomize: (bool) if the dataset should be shuffled
    :param verbose: (int) Verbosity
    """
    def __init__(self, expert_path, train_fraction=0.7, batch_size=64,
                 traj_limitation=-1, randomize=True, verbose=1):
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
        if len(observations.shape) > 2:
            observations = np.reshape(observations, [-1, np.prod(observations.shape[1:])])
        if len(actions.shape) > 2:
            actions = np.reshape(actions, [-1, np.prod(actions.shape[1:])])

        indices = np.random.permutation(len(observations)).astype(np.int64)
        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + batch_size]))
                         for start_idx in range(0, len(indices) - batch_size + 1, batch_size)]

        minibatchlist = np.array(minibatchlist)
        # Number of minibatches used for training
        n_train_batches = np.round(train_fraction * len(minibatchlist)).astype(np.int64)
        minibatches_indices = np.random.permutation(len(minibatchlist))
        # Train/Validation split when using behavior cloning
        train_indices = minibatches_indices[:n_train_batches]
        val_indices = minibatches_indices[n_train_batches:]
        minibatchlist_train = minibatchlist[train_indices]
        minibatchlist_val = minibatchlist[val_indices]

        assert len(train_indices) > 0
        assert len(val_indices) > 0

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
        self.minibatchlist = minibatchlist

        self.dataloader = None
        self.train_loader = DataLoader(minibatchlist_train, self.observations, self.actions,
                                       shuffle=self.randomize, start_process=False)
        self.val_loader = DataLoader(minibatchlist_val, self.observations, self.actions,
                                     shuffle=self.randomize, start_process=False)

        if self.verbose >= 1:
            self.log_info()

    def init_dataloader(self, batch_size):
        """
        :param batch_size: (int)
        """
        indices = np.random.permutation(len(self.observations)).astype(np.int64)
        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + batch_size]))
                         for start_idx in range(0, len(indices) - batch_size + 1, batch_size)]

        # TODO: add keyword allow_partial batch
        # equivalent to batch_size > len(observations)
        if len(minibatchlist) == 0:
            minibatchlist = [np.arange(len(self.observations)).astype(np.int64)]
        self.dataloader = DataLoader(minibatchlist, self.observations, self.actions,
                                     shuffle=self.randomize, start_process=False)

    def __del__(self):
        del self.dataloader, self.train_loader, self.val_loader

    def prepare_pickling(self):
        """
        Exit processes in order to pickle the dataset.
        """
        self.dataloader, self.train_loader, self.val_loader = None, None, None

    def log_info(self):
        """
        Log the information of the dataset
        """
        logger.log("Total trajectories: {}".format(self.num_traj))
        logger.log("Total transitions: {}".format(self.num_transition))
        logger.log("Average returns: {}".format(self.avg_ret))
        logger.log("Std for returns: {}".format(self.std_ret))

    def get_next_batch(self, split=None):
        """
        Get the batch from the dataset

        :param split: (str) the type of data split (can be None, 'train', 'val')
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        dataloader = {
            None: self.dataloader,
            'train': self.train_loader,
            'val': self.val_loader
        }[split]

        if dataloader.process is None:
            dataloader.start_process()
        try:
            return next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            return next(dataloader)

    def plot(self):
        """
        Show histogram plotting of the episode returns
        """
        plt.hist(self.returns)
        plt.show()


class DataLoader(object):
    """
    A custom dataloader to preprocessing observations (including images)
    and feed them to the network.

    Original code for the dataloader from https://github.com/araffin/robotics-rl-srl
    (MIT licence)
    Authors: Antonin Raffin, René Traoré, Ashley Hill

    :param minibatchlist: ([np.ndarray]) list of observations indices (grouped per minibatch)
    :param observations: (np.ndarray) observations or images path
    :param actions: (np.ndarray) actions
    :param n_workers: (int) number of preprocessing worker (for loading the images)
    :param infinite_loop: (bool) whether to have an iterator that can be resetted
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param shuffle: (bool) Shuffle the minibatch after each epoch
    :param start_process: (bool) Start the preprocessing process (default: True)
    :param backend: (str) joblib backend (one of 'multiprocessing', 'sequential', 'threading'
        or 'loky' in newest versions)
    :param sequential: (bool) Do not use subprocess to preprocess the data
        (slower but use less memory for the CI)
    """
    def __init__(self, minibatchlist, observations, actions, n_workers=1,
                 infinite_loop=True, max_queue_len=1, shuffle=False,
                 start_process=True, backend='threading', sequential=True):
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.observations = observations
        self.actions = actions
        self.shuffle = shuffle
        self.queue = Queue(max_queue_len)
        self.process = None
        self.load_images = isinstance(observations[0], str)
        self.backend = backend
        self.sequential = sequential
        self.indices = None
        self.current_minibatch_idx = 0
        if start_process:
            self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        # Skip if in sequential mode
        if self.sequential:
            return
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def sequential_next(self):
        """
        Sequential version of the pre-processing.
        """
        if self.current_minibatch_idx == len(self):
            raise StopIteration

        if self.current_minibatch_idx == 0:
            if self.shuffle:
                self.indices = np.random.permutation(self.n_minibatches).astype(np.int64)
            else:
                self.indices = np.arange(len(self.minibatchlist), dtype=np.int64)

        obs = self.observations[self.minibatchlist[self.current_minibatch_idx]]
        if self.load_images:
            obs = [self._make_batch_element(image_path)
                   for image_path in obs]
            obs = np.concatenate(obs, axis=0)

        actions = self.actions[self.minibatchlist[self.current_minibatch_idx]]
        self.current_minibatch_idx += 1
        return obs, actions

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend=self.backend) as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:

                    obs = self.observations[self.minibatchlist[minibatch_idx]]
                    if self.load_images:
                        if self.n_workers <= 1:
                            obs = [self._make_batch_element(image_path)
                                   for image_path in obs]

                        else:
                            obs = parallel(delayed(self._make_batch_element)(image_path)
                                           for image_path in obs)

                        obs = np.concatenate(obs, axis=0)

                    actions = self.actions[self.minibatchlist[minibatch_idx]]

                    self.queue.put((obs, actions))

                    # Free memory
                    del obs

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, image_path):
        """
        Process one element.

        :param image_path: (str) path to an image
        :return: (np.ndarray)
        """
        # cv2.IMREAD_UNCHANGED is needed to load
        # grey and RGBa images
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Grey image
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if image is None:
            raise ValueError("Tried to load {}, but it was not found".format(image_path))
        # Convert from BGR to RGB
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((1,) + image.shape)
        return image

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        self.current_minibatch_idx = 0
        return self

    def __next__(self):
        if self.sequential:
            return self.sequential_next()

        if self.process is None:
            raise ValueError("You must call .start_process() before using the dataloader")
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()
