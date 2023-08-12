import gymnasium as gym
import random
import numpy as np


def main():
    env = gym.make("Taxi-v3", render_mode="human")

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

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)


    state = env.reset()
    done = False
    rewards = 0

    # this loop is for the animation so you can visually see
    # how the agent is performing.
    for s in range(STEPS_PER_EPISODE):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        # exploit a known action, we'll only used the
        # exploitation since the agent is aleady trained
        if(isinstance(state, int)):
            state = state
        else:
            state = state[0]
        action = np.argmax(qtable[state,:])
        # take the action in the environment
        values= env.step(action)
        print(values)
        new_state = values[0]
        reward = values[1]
        done = values[2]
        info = values[4]
        # update reward
        rewards += reward
        # update the screenshot of the environment
        env.render()

        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

    env.close()


if __name__ == "__main__":
    main()