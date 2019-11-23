import numpy as np
import gym
from gym import ObservationWrapper
from gym.spaces import MultiDiscrete

import matplotlib.pyplot as plt
from matplotlib import animation

class DiscreteQLearningAgent:
    def __init__(self, state_shape, num_of_actions, reward_decay):
        self.q_table = np.zeros((*state_shape, num_of_actions))
        self.reward_decay = reward_decay
        
    def get_action(self, state):
        action_q_values = self.q_table[(*state,)]
        best_action = np.argmax(action_q_values)
        
        return best_action

    def update_table(self, state, action, reward, new_state):
        max_q_value = np.max(self.q_table[(*new_state,)])
        self.q_table[(*state, action)] = reward + self.reward_decay * max_q_value

class MountainCarDiscretizeWrapper(ObservationWrapper):
    def __init__(self, env, num_pos_buckets, num_speed_buckets):
        super().__init__(env)
        self.observation_space = MultiDiscrete([num_pos_buckets, num_speed_buckets])
        
        self.pos_buckets = np.linspace(-1.2, 0.6, num_pos_buckets)
        self.speed_buckets = np.linspace(-0.07, 0.07, num_speed_buckets)

    def observation(self, obs):
        pos, speed = obs
        pos_bucket = np.digitize(pos, self.pos_buckets)
        speed_bucket = np.digitize(speed, self.speed_buckets)
        
        return [pos_bucket, speed_bucket]
        
    
def train_agent(agent, env, episodes):
    for i in range(episodes):
        state = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            
            # After every step update our q table
            agent.update_table(state, action, reward, new_state)
            
            # Set our state variable
            state = new_state
        
        print(i, ": ", step, "steps")
        
def test_agent(agent, env, episodes):
    for i in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            
def generate_episode_gif(agent, env, filepath):
    frames = []
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        frames.append(env.render(mode='rgb_array'))
        
    patch = plt.imshow(frames[0])
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filepath, writer='imagemagick', fps=60)
            
def visualize_value_function(agent, num_pos_buckets, num_speed_buckets):
    arr = np.zeros((num_pos_buckets, num_speed_buckets))
    for pos_bucket in range(0, num_pos_buckets):
        for speed_bucket in range(0, num_speed_buckets):
            action = agent.get_action([pos_bucket, speed_bucket])
            state_value = agent.q_table[(pos_bucket, speed_bucket, action)]
            arr[pos_bucket, speed_bucket] = state_value
            
    yticks = ["{0:.2f}".format(value) for value in np.linspace(-1.2, 0.6, num_pos_buckets)]
    xticks = ["{0:.2f}".format(value) for value in np.linspace(-0.07, 0.07, num_speed_buckets)]        
    
    plt.imshow(arr, vmin=np.min(arr), vmax=0, cmap='gist_heat', aspect='auto')
    plt.colorbar()
    
    plt.xticks(np.arange(0, num_speed_buckets), xticks, rotation='vertical')
    plt.yticks(np.arange(0, num_pos_buckets), yticks)
    plt.ylabel("Position")
    plt.xlabel("Speed")
    
if __name__ == "__main__":
    
    NUM_POS_BUCKETS = 50
    NUM_SPEED_BUCKETS = 50
    
    env = gym.make("MountainCar-v0").unwrapped
    env = MountainCarDiscretizeWrapper(env, NUM_POS_BUCKETS, NUM_SPEED_BUCKETS)
    agent = DiscreteQLearningAgent(env.observation_space.nvec, env.action_space.n, 0.99)
    
    train_agent(agent, env, 1000)
    env.close()
    
    env = gym.make("MountainCar-v0").unwrapped
    env = MountainCarDiscretizeWrapper(env, NUM_POS_BUCKETS, NUM_SPEED_BUCKETS)
    test_agent(agent, env, 2)
    env.close()
    
    visualize_value_function(agent, NUM_POS_BUCKETS, NUM_SPEED_BUCKETS)