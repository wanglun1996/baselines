import os

from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.time_limit import TimeLimit
from gym import spaces
import numpy as np
import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO2, bench
from stable_baselines.common.policies import MlpLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2.ppo2 import safe_mean


class CustomLSTMPolicy1(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, net_arch=[8, 'lstm', 8],
                         layer_norm=False, feature_extraction="mlp", **_kwargs)


class CustomLSTMPolicy2(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=['lstm', 8], layer_norm=True, feature_extraction="mlp", **_kwargs)


class CustomLSTMPolicy3(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm'], layer_norm=False, feature_extraction="mlp", **_kwargs)


class CustomLSTMPolicy4(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


def _pos_obs(full_obs):
    xpos, _xvel, thetapos, _thetavel = full_obs
    return xpos, thetapos


class CartPoleNoVelEnv(CartPoleEnv):
    """Variant of CartPoleEnv with velocity information removed. This task requires memory to solve."""

    def __init__(self):
        super().__init__()
        idxs = [0, 2]
        low = self.observation_space.low[idxs]
        high = self.observation_space.high[idxs]
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        full_obs = super().reset()
        return _pos_obs(full_obs)

    def step(self, action):
        full_obs, rew, done, info = super().step(action)
        return _pos_obs(full_obs), rew, done, info


N_TRIALS = 100
NUM_ENVS = 16

MODELS = [A2C, ACER, ACKTR, PPO2]
LSTM_POLICIES = [MlpLstmPolicy, CustomLSTMPolicy1, CustomLSTMPolicy2, CustomLSTMPolicy3, CustomLSTMPolicy4]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("policy", LSTM_POLICIES)
def test_lstm_policy(model_class, policy):
    try:
        # create and train
        if model_class == PPO2:
            model = model_class(policy, 'CartPole-v1', nminibatches=1)
        else:
            model = model_class(policy, 'CartPole-v1')
        model.learn(total_timesteps=100, seed=0)

        env = model.get_env()
        # predict and measure the acc reward
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
        # saving
        model.save("./test_model")
        del model, env
        # loading
        _ = model_class.load("./test_model", policy=policy)

    finally:
        if os.path.exists("./test_model"):
            os.remove("./test_model")


@pytest.mark.expensive
def test_lstm_train():
    """Test that LSTM models are able to achieve >=450 (out of 500) reward on CartPoleNoVelEnv.

    This environment requires memory to perform well in."""
    def make_env(i):
        env = CartPoleNoVelEnv()
        env = TimeLimit(env, max_episode_steps=500)
        env = bench.Monitor(env, None, allow_early_resets=True)
        env.seed(i)
        return env

    env = SubprocVecEnv([lambda: make_env(i) for i in range(NUM_ENVS)])
    env = VecNormalize(env)
    model = PPO2(MlpLstmPolicy, env, n_steps=128, nminibatches=NUM_ENVS, lam=0.95, gamma=0.99,
                 noptepochs=10, ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, verbose=1)

    eprewmeans = []
    def reward_callback(local, _):
        nonlocal eprewmeans
        eprewmeans.append(safe_mean([ep_info['r'] for ep_info in local['ep_info_buf']]))

    model.learn(total_timesteps=100000, seed=0, callback=reward_callback)

    # Maximum episode reward is 500.
    # In CartPole-v1, a non-recurrent policy can easily get >= 450.
    # In CartPoleNoVelEnv, a non-recurrent policy doesn't get more than ~50.
    # LSTM policies can reach above 400, but it varies a lot between runs; consistently get >=150.
    # See PR #244 for more detailed benchmarks.

    average_reward = sum(eprewmeans[-4:]) / 4
    assert average_reward >= 150, "Mean reward below 200; per-episode rewards {}".format(average_reward)
