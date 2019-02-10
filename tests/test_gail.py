import gym

from stable_baselines import GAIL
from stable_baselines.gail.dataset.mujocodataset import MujocoDataset

EXPERT_PATH = "stable_baselines/gail/dataset/expert_pendulum.npz"


def test_gail():
    env = gym.make('Pendulum-v0')
    dataset = MujocoDataset(expert_path=EXPERT_PATH, traj_limitation=-1, verbose=1)

    model = GAIL('MlpPolicy', env,
                 expert_dataset=dataset, hidden_size_adversary=64, verbose=1)

    model.learn(1000)

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(obs)
        if done:
            obs = env.reset()
