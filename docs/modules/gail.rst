.. _gail:

.. automodule:: stable_baselines.gail


GAIL
====

`Generative Adversarial Imitation Learning (GAIL) <https://arxiv.org/abs/1606.03476>`_


Notes
-----

- Original paper: https://arxiv.org/abs/1606.03476

If you want to train an imitation learning agent
------------------------------------------------


Step 1: Generate expert data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Step 2: Run GAIL
~~~~~~~~~~~~~~~~


**In case you want to run Behavior Cloning (BC)**

Use the ``.pretrain`` method.


**Others**

Thanks to the open source:

-  @openai/imitation
-  @carpedm20/deep-rl-tensorflow


Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️ (using MPI)
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Parameters
----------

.. autoclass:: GAIL
  :members:
  :inherited-members:
