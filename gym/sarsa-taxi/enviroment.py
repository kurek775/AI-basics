
import gymnasium as gym
import numpy as np
import time
from agent import SARSA_Learning


env = gym.make("Taxi-v3")
env.reset()

env.observation_space

env.action_space
# actions:
# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP


# define the parameters

# step size
alpha = 0.1
# discount rate
gamma = 0.9
# epsilon-greedy parameter
epsilon = 0.2
# number of simulation episodes
numberEpisodes = 10000

# initialize
SARSA1 = SARSA_Learning(env, alpha, gamma, epsilon, numberEpisodes)
# simulate
SARSA1.simulateEpisodes()
# compute the final policy
SARSA1.computeFinalPolicy()

# extract the final policy
finalLearnedPolicy = SARSA1.learnedPolicy

# simulate the learned policy for verification
while True:
    # to interpret the final learned policy you need this information
    # actions: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    # let us simulate the learned policy
    # this will reset the environment and return the agent to the initial state
    env = gym.make("Taxi-v3", render_mode="human")
    (currentState, prob) = env.reset()
    env.render()
    time.sleep(2)
    # since the initial state is not a terminal state, set this flag to false
    terminalState = False
    for i in range(100):
        # here we step and return the state, reward, and boolean denoting if the state is a terminal state
        if not terminalState:
            (currentState, currentReward, terminalState, _, _) = env.step(
                int(finalLearnedPolicy[currentState]))
            time.sleep(1)
        else:
            break
    time.sleep(0.5)
env.close()
