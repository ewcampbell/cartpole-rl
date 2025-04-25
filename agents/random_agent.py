from envs.cartpole_env import CartPoleEnv
import random

env = CartPoleEnv()
state = env.reset()

for t in range(100):
    action = random.choice([0, 1])  # Randomly pick left or right
    next_state, reward, done = env.step(action)
    env.render()
    if done:
        print(f"Episode ended after {t+1} timesteps.")
        break
