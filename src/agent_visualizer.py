from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
import numpy as np

class ReacherAgentVisualizer:
    def __init__(self, file_path: str = 'Reacher_Linux/Reacher.x86_64',
                 actor_path: str = 'results/checkpoints/checkpoint_actor_Final_Agent.pth',
                 critic_path: str = 'results/checkpoints/checkpoint_critic_Final_Agent.pth',
                 seed: int = 2):

        self.env = UnityEnvironment(file_name=file_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Reset to get environment specs
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.state_size = env_info.vector_observations.shape[1]
        self.action_size = self.brain.vector_action_space_size

        # Initialize agent and load weights
        self.agent = Agent(state_size=self.state_size, action_size=self.action_size, random_seed=seed)
        self.agent.actor_local.load_state_dict(torch.load(actor_path))
        self.agent.critic_local.load_state_dict(torch.load(critic_path))

    def run(self, max_t: int = 1000):
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)

        for t in range(max_t):
            actions = self.agent.act_probabilistic(states, add_noise=False)
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states = next_states
            scores += rewards
            if np.any(dones):
                break

        avg_score = np.mean(scores)
        print(f"Episode finished. Average Score across agents: {avg_score:.2f}")
        self.env.close()

if __name__ == "__main__":
    visualizer = ReacherAgentVisualizer(file_path = 'Reacher_Linux/Reacher.x86_64',
                 actor_path = 'results/checkpoints/checkpoint_actor_Final_Agent_03.pth',
                 critic_path = 'results/checkpoints/checkpoint_critic_Final_Agent_03.pth',
                 seed = 2)
    visualizer.run(max_t = 2000)
