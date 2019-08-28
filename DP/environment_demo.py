import random
import numpy as np
from environment import Environment


class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    #Total Number Of Episode
    NumEp=5
    # Try NumEp times game.
    stepList=[]
    rewardList=[]
    
    for i in range(NumEp):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False
        # 追加 print
        print("-----------PROCESS------------------")
        print("START State:",state)
        # 追加 変数step
        step=0


        while not done:
            action = agent.policy(state)
            #追加 print
            print(action)
            next_state, reward, done = env.step(action)
            #追加print
            print("Next:",next_state)
            total_reward += reward
            state = next_state
            step+=1
        print("------------------------RESULT-----------------------")
        #総step
        print("Step Sum =",step)
        print("Episode {}: Agent gets {} reward.".format(i, total_reward))
        stepList.append(step)
        rewardList.append(total_reward)
    print("\n\n ***************TOTAL RESULT*********************\n\n")
    print("Average Step=",np.average(stepList))
    print("Std of Step=",np.std(stepList))
    print("StepList:",stepList)
    print("\n")
    np.set_printoptions(precision=2)
    rewardList=np.array(rewardList)
    print("Average Reward=",np.average(rewardList))
    print("Std of Reward=",np.std(rewardList))
    print("RewardList:",rewardList)


if __name__ == "__main__":
    main()
