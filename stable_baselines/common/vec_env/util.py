"""
Helpers for dealing with vectorized environments.
"""

from collections import OrderedDict

import gym
import numpy as np


def copy_obs_dict(obs):
    """
    Deep-copy a dict of numpy arrays.

    :param obs: (dict<ndarray>): a dict of numpy arrays.
    :return (dict<ndarray>) a dict of copied numpy arrays.
    """
    return {k: np.copy(v) for k, v in obs.items()}


def obs_to_dict(obs):
    """
    Convert an observation into a dict.

    :param obs: (gym.spaces.Space) an observation space.
    :return: (dict<ndarray>) if obs was a dictionary, returns obs.
             Otherwise, returns a dictionary with a single key None and value obs.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}


def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if singleton.

    :param obs_dict: (dict<ndarray>) a dict of numpy arrays.
    :return (ndarray or dict<ndarray>): if obs_dict has a single element
            with key None, returns that value. Otherwise, returns the original dict.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.

    :param obs_space: (gym.spaces.Space) an observation space
    :return (tuple) A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes
