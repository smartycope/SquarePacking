from gymnasium.envs.registration import register
# from gym.envs.registration import register
# from SquareEnv import SquareEnv

register(
     id="SquarePacking/Square-v0",
     entry_point="SquarePacking.env:SquareEnv",
     max_episode_steps=500,
     autoreset=False,
     order_enforce=True,
)
