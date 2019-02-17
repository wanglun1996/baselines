"""
The code is used to train BC imitator, or pretrained GAIL imitator
"""
import argparse

import tensorflow as tf
from tqdm import tqdm

from stable_baselines import logger
from stable_baselines.common import set_global_seeds, tf_util
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.gail import GAIL
from stable_baselines.gail.dataset.mujocodataset import MujocoDataset


def argsparser():
    """
    make a behavior cloning argument parser

    :return: (ArgumentParser)
    """
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env', help='environment ID', required=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert-path', type=str, required=True)
    parser.add_argument('--traj-limitation', type=int, default=-1)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('-iters', '--n-iters', help='Max iteration for training BC', type=int, default=1e5)
    return parser


def learn(model, dataset, optim_batch_size=128, max_iters=1e4, adam_epsilon=1e-5, optim_stepsize=3e-4,
          verbose=False):
    """
    Learn a behavior clone policy, and return the save location

    :param dataset: (Dset or MujocoDset) the dataset manager
    :param optim_batch_size: (int) the batch size
    :param max_iters: (int) the maximum number of iterations
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param optim_stepsize: (float) the optimizer stepsize
    :param verbose: (bool)
    :return: (str) the save location for the TensorFlow model
    """

    val_per_iter = int(max_iters / 10)

    with model.trpo.graph.as_default():
        policy = model.trpo.policy_pi
        # placeholder
        obs_ph = policy.obs_ph
        action_ph = policy.pdtype.sample_placeholder([None])
        loss = tf.reduce_mean(tf.square(action_ph - policy.deterministic_action))
        var_list = model.trpo.params
        adam = MpiAdam(var_list, epsilon=adam_epsilon)
        lossandgrad = tf_util.function([obs_ph, action_ph], [loss] + [tf_util.flatgrad(loss, var_list)])

        # tf_util.initialize()
        adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        expert_obs, expert_actions = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, grad = lossandgrad(expert_obs, expert_actions)
        adam.update(grad, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            expert_obs, expert_actions = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(expert_obs, expert_actions)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

    return model


def main(args):
    """
    start training the model

    :param args: (ArgumentParser) the training argument
    """
    set_global_seeds(args.seed)

    dataset = MujocoDataset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    model = GAIL("MlpPolicy", args.env, dataset)

    with model.trpo.sess.as_default():
        model = learn(model, dataset, max_iters=args.n_iters, verbose=args.verbose)

    return model


if __name__ == '__main__':
    parser = argsparser()
    main(parser.parse_args())
