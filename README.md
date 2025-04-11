Given a grid-worl/robot navigation problem (see map below):

----------   R=robot, O=obstacle, G=goal
|R    OO |
|     OO |
|O       |
|        |
|    O   |
|  OOO   |
|        |
|  O    G|
----------

.. implement a very simple Q-learning algorithm or - alternatively - a very simple
policy gradient algorithm and try to run it and have it learn how to act
optimally in the environment. In case you chose the policy gradient solution,
you won't need to implement an extra value function (also no value function loss
term, just use the raw environment returns).

You only need to change the `run_experiment.py` script. The code in the models.py
and room_env.py file does NOT need to be altered. Use it for reference, only.

Also:
- Chose a deep learning framework (torch preferred, but tf also ok).
- Use the already provided environment class `RoomEnv` in the `room_env.py`
  file and one of the given model classes (`MyTorchModel`, `MyKerasModel`)
  in the `model.py` file.
