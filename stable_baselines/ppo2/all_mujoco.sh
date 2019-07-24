#!/bin/bash

ENVS="CartPole-v0 Acrobot-v1 MountainCar-v0 Reacher-v2 HalfCheetah-v2 Hopper-v2 Ant-v2 Humanoid-v2"
NUM_JOBS=6

parallel -j ${NUM_JOBS} --header : --results outputs/parallel python -m stable_baselines.ppo2.run_mujoco --env {env} --seed {seed} ::: env ${ENVS} ::: seed 0 1 2
