import gym
import pytest

from stable_baselines import A2C, GAIL, PPO1, PPO2, TRPO
from stable_baselines.gail.dataset.dataset import MujocoDataset
from stable_baselines.gail.dataset.record_expert import train_pendulum_expert

EXPERT_PATH = "stable_baselines/gail/dataset/expert_pendulum.npz"


def test_gail():
    env = gym.make('Pendulum-v0')
    dataset = MujocoDataset(expert_path=EXPERT_PATH, traj_limitation=10, verbose=1)

    # Note: train for 1M steps to have a working policy
    model = GAIL('MlpPolicy', env, adversary_entcoeff=0.0, lam=0.92, max_kl=0.001,
                 expert_dataset=dataset, hidden_size_adversary=64, verbose=1)

    model.learn(1000)
    model.save("GAIL-Pendulum")
    model = model.load("GAIL-Pendulum", env=env)

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()


def test_generate_expert_data():
    train_pendulum_expert(n_timesteps=1000, n_episodes=10)


@pytest.mark.parametrize("model_class", [A2C, GAIL, PPO1, PPO2, TRPO])
def test_behavior_cloning(model_class):
    dataset = MujocoDataset(expert_path=EXPERT_PATH, traj_limitation=10)
    if model_class == GAIL:
        model = model_class("MlpPolicy", "Pendulum-v0", dataset)
    else:
        model = model_class("MlpPolicy", "Pendulum-v0")
    model.pretrain(dataset, num_iter=1000)
    model.save("test-pretrain")
