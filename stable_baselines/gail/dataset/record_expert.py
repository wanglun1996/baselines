import numpy as np
from gym import spaces


def generate_expert_traj(model, save_path, n_timesteps=60000, n_episodes=100):
    """
    Train expert controller (if needed) and record expert trajectories.
    NOTE: only Box and Discrete spaces are supported for now.

    :param model: (RL model)
    :param save_path: (str)
    :param n_timesteps: (int)
    :param n_episodes: (int)
    """
    env = model.get_env()

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    if n_timesteps > 0:
        model.learn(n_timesteps)

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    while ep_idx < n_episodes:
        observations.append(obs)
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        if done:
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    np.savez(save_path, **numpy_dict)
