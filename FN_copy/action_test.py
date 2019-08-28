import retro
import gym
from gym.spaces import MultiBinary as mb
import numpy as np
import random

def main():
    env = retro.make(game='Pong-Atari2600', players=1)
    env_gym=gym.make("CartPole-v0")
    obs = env.reset()
    obs_gym=env_gym.reset()

    a=env.action_space.sample()
    print("\n-------------------------------------\n")
    print("env.action_space")
    print(env.action_space)
    print("\nenv.action_space.sample")
    print(env.action_space.sample())
    print(a)
    print(a.argmax())
    print("\nenv_gym.action_space")
    print(env_gym.action_space)
    print("\nenv_gym.action_space.n")
    print(env_gym.action_space.n)
    print("\nenv_gym.action_space.sample")
    print(env_gym.action_space.sample())
    print("\nnumpyの二項分布乱数　numpy array")
    print(np.random.binomial(1,0.5,env.action_space.n))
    print("\nnp.array(range(env.action_space.n))")
    print(np.array(range(env.action_space.n)))
    print("\nlen(np.array(range(env.action_space.n)))")
    print(len(np.array(range(env.action_space.n))))
    print("\nlen(np.array(range(env_gym.action_space.n)))")
    print(len(np.array(range(env_gym.action_space.n))))
    print("\nlen(np.array(range(env.action_space.n)))")
    print(len(np.array(range(env.action_space.n))))

    # while True:
    #     # action_space will by MultiBinary(16) now instead of MultiBinary(8)
    #     # the bottom half of the actions will be for player 1 and the top half for player 2
    #     obs, rew, done, info = env.step(env.action_space.sample())
    #     # rew will be a list of [player_1_rew, player_2_rew]
    #     # done and info will remain the same
    #     env.render()
    #     if done:
    #         obs = env.reset()
    # env.close()


if __name__ == "__main__":
    main()