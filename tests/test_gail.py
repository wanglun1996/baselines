import gym

from stable_baselines import GAIL
from stable_baselines.gail.dataset.mujocodataset import MujocoDataset

EXPERT_PATH = "stable_baselines/gail/dataset/expert_pendulum.npz"


def test_gail():
    env = gym.make('Pendulum-v0')
    dataset = MujocoDataset(expert_path=EXPERT_PATH, traj_limitation=-1, verbose=1)

    # Note: train for 1M steps (4 workers) to have a working policy
    model = GAIL('MlpPolicy', env, adversary_entcoeff=0.0, lam=0.92, max_kl=0.001,
                 expert_dataset=dataset, hidden_size_adversary=64, verbose=1)

    model.learn(1000)
    model.save("GAIL-Pendulum")
    model = model.load("GAIL-Pendulum")

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(obs)
        if done:
            obs = env.reset()
