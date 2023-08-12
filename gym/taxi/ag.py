import gym
import numpy as np
import random

def train_agent():
    env = gym.make("Taxi-v3")

    # initialize the q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # set the number of episodes
    EPISODES = 500
    STEPS_PER_EPISODE = 50
    # hyperparameters
    epsilon = 1.0
    decay_rate= 0.005
    learning_rate = 0.9
    discount_rate = 0.8

    for episode in range(EPISODES):
      # At the beginning of each episode, done is false
        done = False
        # reset the env for each new episodeY
        state = env.reset()
        print('state',state[0])
        if(isinstance(state, int)):
            state = state
        else:
            state = state[0]
        for step in range(STEPS_PER_EPISODE):

            # in here, we have to decide whether to 
            # explore the env or exploit what we already know
            # this is where the exploration-exploitation tradeoff comes to play
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state,:])

            values= env.step(action)
            print(values)
            new_state = values[0]
            reward = values[1]
            done = values[2]
            info = values[4]
            # Q-learning algorithm implementation1
            print('new_state',new_state)
            if(isinstance(state, int)):
                state = state
            else:
                state = state[0]
            print('state',state)
            # state = state[0]
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            state = new_state

            if done: 
                break
        epsilon = np.exp(-decay_rate*episode)

    return qtable

if __name__ == "__main__":
    qtable = train_agent()
    np.save("qtable.npy", qtable)  # save the trained qtable to a file
