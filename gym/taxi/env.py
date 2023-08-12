import gym
import numpy as np

def visualize_agent(qtable):
    env = gym.make("Taxi-v3", render_mode="human")

    state = env.reset()
    done = False
    rewards = 0

    STEPS_PER_EPISODE = 50

    # this loop is for the animation so you can visually see
    # how the agent is performing.
    for s in range(STEPS_PER_EPISODE):
        env.render()
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
 

        print(f"score: {rewards}")
        state = new_state
        env.render()
        if done == True:
            break

    env.close()

if __name__ == "__main__":
    qtable = np.load("qtable.npy")  # load the trained qtable from the file
    visualize_agent(qtable)
