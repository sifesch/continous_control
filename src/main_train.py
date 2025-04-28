from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from utils import create_complex_training_plot, create_simple_training_plot
from ddpg_agent import Agent
import numpy as np
from config_loader import Configurations

class DDPGTrainer:
    def __init__(self, config:Configurations, file_path:str = 'Reacher_Linux/Reacher.x86_64'):

        self.file_path = file_path
        self.noise_choice = config.noise_config.GENERAL.PROB_NOISE_OR_OU
        self.noise_decay_choice = config.noise_config.GENERAL.ACT_NOISE_DECAY
        self.n_episodes = config.hyperparameters.N_EPISODES
        self.max_t = config.hyperparameters.MAX_T
        self.trialname = config.hyperparameters.TRIAL_NAME

        self.initialize_unity()
        self.print_environment_information()
        self.agent = Agent(state_size=self.state_size, action_size=self.action_size, random_seed=2)

    def initialize_unity(self) -> None:
        self.env = UnityEnvironment(file_name=self.file_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        self.states = self.env_info.vector_observations
        self.state_size = self.states.shape[1]

    def ddpg_train(self, print_every:int = 10, target_score: int = 32):
        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, self.n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations                   
            if self.noise_choice == 'ou':
                self.agent.noise.reset()                                            
            score = np.zeros(self.num_agents)                             
            self.agent.episode = i_episode

            for t in range(self.max_t):
                if self.noise_choice == 'prob':
                    actions = self.agent.act_probabilistic(states)
                elif self.noise_choice == 'ou':
                    actions = self.agent.act_ou(states)

                actions = np.clip(actions, -1, 1)                     
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # loop through each agent and perform step
                for i in range(self.num_agents):
                    self.agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i], t=t)

                states = next_states
                score += rewards

                if np.any(dones):
                    if self.noise_choice == 'prob' and self.noise_decay_choice == True:
                        self.agent.noise_update()
                    if self.noise_choice == 'ou' and self.noise_decay_choice  == True:
                        self.agent.noise.decay_sigma()
                    break

            avg_score = np.mean(score)
            scores.append(avg_score)
            scores_deque.append(avg_score)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score), end="")

            if i_episode % 10 == 0:
                torch.save(self.agent.actor_local.state_dict(), f'results/checkpoints/checkpoint_actor_{self.trialname}.pth')
                torch.save(self.agent.critic_local.state_dict(), f'results/checkpoints/checkpoint_critic_{self.trialname}.pth')

            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
            if np.mean(scores_deque) >= target_score:  
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, avg_score))
                torch.save(self.agent.actor_local.state_dict(), f'results/checkpoints/checkpoint_actor_{self.trialname}.pth')
                torch.save(self.agent.critic_local.state_dict(), f'results/checkpoints/checkpoint_critic_{self.trialname}.pth')
                break

        return scores
    
    def print_environment_information(self) -> None:
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('There are {} agents. Each observes a state with length: {}'.format(self.states.shape[0], self.state_size))
        print('The state for the first agent looks like:', self.states[0])

    def genereate_result_plots(self,scores) -> None:
        create_simple_training_plot(scores, trialname=self.trialname)
        create_complex_training_plot(scores, trialname=self.trialname, window_size=100)
        config.save_config(output_path=f'results/configurations/trial_configurations_{self.trialname}.yaml')

if __name__ == '__main__':
    config = Configurations(config_path = 'config/train_config.yaml')
    trainer = DDPGTrainer(config=config)
    scores = trainer.ddpg_train()
    trainer.genereate_result_plots(scores=scores)
    trainer.env.close()

    


