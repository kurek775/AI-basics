import gymnasium as gym
env = gym.make("Taxi-v3", render_mode="human")
observation, info = env.reset(seed=42)
state = env.unwrapped.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)

print(state)

# get the whole reward table
print(env.unwrapped.P)

# get the reward table for a specific statte
print(env.unwrapped.P[state])
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()