.. _pretrain:

.. automodule:: stable_baselines.gail


Pre-Training (Behavior Cloning)
===============================

Behavior Cloning (BC) treats the problem of imitation learning, i.e., using expert demonstrations, as a supervised learning problem.
That is to say, given expert trajectories (observations-actions pairs), the policy network is trained to reproduce the expert behavior:
for the same observation, the action taken by the policy must be the same of the expert.

This method can used to pretrain the RL model and therefore accelerate training.


.. note::

	Only ``Box`` and ``Discrete`` spaces are supported for now for pre-training a model.


.. note::

  Images datasets are treated a bit differently as other datasets to avoid memory issues.
  The images from the expert demonstrations must be located in a folder, not in the expert numpy archive.



Generate Expert Trajectories
----------------------------

.. code-block:: python

  from stable_baselines import DQN
  from stable_baselines.gail import generate_expert_traj

  model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
  generate_expert_traj(model, 'expert_cartpole', n_timesteps=int(1e5), n_episodes=10)



Pre-Train a Model using Behavior Cloning
----------------------------------------

Using the ``expert_cartpole.npz`` dataset generated using the previous script.

.. code-block:: python

  from stable_baselines import PPO2
  from stable_baselines.gail import ExpertDataset
  # Using only one expert trajectory
  dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                          traj_limitation=1, batch_size=128)

  model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
  # Pretrain the PPO2 model
  model.pretrain(dataset, num_iter=10000)

  # Test the pre-trained model
  env = model.get_env()
  obs = env.reset()

  reward_sum = 0.0
  for _ in range(1000):
  	action, _ = model.predict(obs)
  	obs, reward, done, _ = env.step(action)
  	reward_sum += reward
  	env.render()
  	if done:
  		print(reward_sum)
  		reward_sum = 0.0
  		obs = env.reset()

  env.close()


Data Structure of the Expert Dataset
------------------------------------

The expert dataset is a ``.npz`` archive. The data is save in python dictionary format with keys: ``actions``, ``episode_returns``, ``rewards``, ``obs``,
``episode_starts``.

In case of images, ``obs`` contains the relative path to the images.

obs, actions: shape (N * L, ) + S

where N = # episodes, L = episode length
and S is the environment observation/action space.

S = (1, ) for discrete space


.. autoclass:: ExpertDataset
  :members:
  :inherited-members:


.. autoclass:: DataLoader
  :members:
  :inherited-members:


.. autofunction:: generate_expert_traj
