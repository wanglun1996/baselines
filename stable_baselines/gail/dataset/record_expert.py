"""
Train expert controller and save data.
"""
import numpy as np

from stable_baselines import SAC
# from stable_baselines import DQN

# model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
# model.learn(int(1e5))

def train_pendulum_expert(n_timesteps=60000, n_episodes=100):
    """
    :param n_timesteps: (int)
    """
    model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
    model.learn(n_timesteps)

    env = model.get_env()
    max_steps = 200
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actions = np.zeros((n_episodes, max_steps, n_actions))
    observations = np.zeros((n_episodes, max_steps, n_features))
    rewards = np.zeros((n_episodes, max_steps))
    episode_returns = np.zeros((n_episodes,))

    ep_idx = 0
    step = 0
    obs = env.reset()
    while ep_idx < n_episodes:
        observations[ep_idx, step] = obs
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        actions[ep_idx, step] = action
        rewards[ep_idx, step] = reward
        step += 1
        if done:
            obs = env.reset()
            episode_returns[ep_idx] = rewards[ep_idx, :].sum()
            assert step == max_steps
            ep_idx += 1
            step = 0


    numpy_dict = {
        # Old format
        'acs': actions,
        'rews': actions,
        'ep_rets': episode_returns,
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    np.savez("expert_pendulum", **numpy_dict)
