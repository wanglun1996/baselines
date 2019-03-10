import gym
import pytest

from stable_baselines import A2C, ACER, ACKTR, GAIL, DDPG, DQN, PPO1, PPO2, TRPO, SAC
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.gail import ExpertDataset, generate_expert_traj

EXPERT_PATH = "stable_baselines/gail/dataset/expert_pendulum.npz"
EXPERT_PATH_DISCRETE = "stable_baselines/gail/dataset/expert_cartpole.npz"


def test_gail():
    env = gym.make('Pendulum-v0')
    dataset = ExpertDataset(expert_path=EXPERT_PATH, traj_limitation=10, verbose=1)

    # Note: train for 1M steps to have a working policy
    model = GAIL('MlpPolicy', env, adversary_entcoeff=0.0, lam=0.92, max_kl=0.001,
                 expert_dataset=dataset, hidden_size_adversary=64, verbose=1)

    model.learn(1000)
    model.save("GAIL-Pendulum")
    model = model.load("GAIL-Pendulum", env=env)
    model.learn(1000)

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    del dataset, model


def test_generate_pendulum():
    model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
    generate_expert_traj(model, 'expert_pendulum', n_timesteps=1000, n_episodes=10)


def test_generate_cartpole():
    model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
    generate_expert_traj(model, 'expert_cartpole', n_timesteps=1000, n_episodes=10)


# @pytest.mark.parametrize("model_class", [A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO])
# def test_pretrain_images():
#     env = make_atari_env("PongNoFrameskip-v4", num_env=1, seed=0)
#     env = VecFrameStack(env, n_stack=4)
#     model = PPO2('CnnPolicy', env)
#     generate_expert_traj(model, 'expert_pong', n_timesteps=0, n_episodes=1,
#                          image_folder='/tmp/recorded_images/')
#
#     expert_path = 'expert_pong.npz'
#     dataset = ExpertDataset(expert_path=expert_path, traj_limitation=1, batch_size=32)
#     model.pretrain(dataset, num_iter=100)
#     del dataset, model


@pytest.mark.parametrize("model_class", [A2C, GAIL, DDPG, PPO1, PPO2, SAC, TRPO])
def test_behavior_cloning_box(model_class):
    """
    Behavior cloning with continuous actions.
    """
    dataset = ExpertDataset(expert_path=EXPERT_PATH, traj_limitation=10)
    if model_class == GAIL:
        model = model_class("MlpPolicy", "Pendulum-v0", dataset)
    else:
        model = model_class("MlpPolicy", "Pendulum-v0")
    model.pretrain(dataset, num_iter=1000)
    model.save("test-pretrain")
    del dataset, model


@pytest.mark.parametrize("model_class", [A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO])
def test_behavior_cloning_discrete(model_class):
    dataset = ExpertDataset(expert_path=EXPERT_PATH_DISCRETE, traj_limitation=10)
    if model_class == GAIL:
        # TODO: discrete actions support for GAIl
        # model = model_class("MlpPolicy", "CartPole-v1", dataset)
        return
    else:
        model = model_class("MlpPolicy", "CartPole-v1")
    model.pretrain(dataset, num_iter=1000)
    model.save("test-pretrain")
    del dataset, model
