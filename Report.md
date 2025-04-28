# Report Introduction

This report contains of three main sections. In the first section [Learning Algorithm](#learning-algorithm) the technicalities are explained. In this section the Model Architecture, DDPG Agent, the Noise Process and the Hyperparameters used for this project are introduced and explained. The second section [Plot of Rewards](#plot-of-rewards) contains visualizations of the scores achieved during training. In the third section [Ideas for Future Work](#ideas-for-future-work) multiple approaches to include in future work on this project are proposed.

- [Report Introduction](#report-introduction)
- [Learning Algorithm](#learning-algorithm)
  - [Deep Deterministic Policy Gradient Algorithm (DDPG)](#deep-deterministic-policy-gradient-algorithm-ddpg)
    - [Experiements and Adjustments that lead to solving the environment](#experiements-and-adjustments-that-lead-to-solving-the-environment)
  - [Model Architecture Actor-Critic Network](#model-architecture-actor-critic-network)
  - [Noise Process](#noise-process)
  - [Chosen Hyperparameters \& brief Explanation of Hyperparameters](#chosen-hyperparameters--brief-explanation-of-hyperparameters)
- [Plot of Rewards](#plot-of-rewards)
- [Ideas for Future Work](#ideas-for-future-work)
- [References](#references)

# Learning Algorithm

## Deep Deterministic Policy Gradient Algorithm (DDPG)

Lillicrap et al. (2015) introduced the DDPG algorithm, which represents a model-free, off-policy reinforcement learning algorithm. The algorithm utilizes an actor-critic approach and Deep Q-Learning to solve continous control problems. The DDPG algorithm can be found in the conference paper of Lillicrap et al. (2015) (https://arxiv.org/pdf/1509.02971).

#TODO ADD DETAILED EXPLANATION OF DDPG

For a visual explanation of the steps of the DDPG, we refer to the Authors Zhou, Huang, and Fränti (2021), who describe the steps of the DDPG in their review of motion planning algorithms (See Figure 1).

![Steps of a DDPG, adapted from Zhou, Huang, and Fränti](https://www.researchgate.net/publication/356554045/figure/fig27/AS:11431281119098631@1675998577919/Steps-of-DDPG-DDPG-combines-the-replay-buffer-actor-critic-architecture-and.png)

*Figure 1: Steps of DDPG. DDPG combines the replay buffer, actor-critic architecture, and deterministic policy. First, action is selected by policy network and reward is obtained. State transits to next state. Second, experience tuple is saved in replay buffer. Third, experiences are sampled from replay buffer for training. Fourth, critic network is updated. Finally, policy network is updated from [Zhou, Huang, and Fränti, 2021, A review of motion planning algorithms for intelligent robots, Journal of Intelligent Manufacturing (2022) 33:387–424, https://doi.org/10.1007/s10845-021-01867-z]*

The implementation of the DDPG algorithm in code was adapted from the previous exercises of the pendulum from the Udacity course (see: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py). Further adjustments were made to the code from the pendulum exercise to solve the environment.

### Experiements and Adjustments that lead to solving the environment

As the agent appeared not to really learn well with different sets of Hyperparameters, new functionalities were added to the DDPG Agent. First, to improve the learning process and avoid that the gradients become too large during backpropagation, gradient clipping was implemented in the learning function. Pascanu, Mikolov, & Bengio (2012) proposed this gradient clipping strategy to deal with exploding gradients and a soft constraint for the vanishing gradients problem. However, this implementation still did not lead to fast learning and an average reward of 30 over 100 consecutive episodes (at least not after the tested number of episodes). Thus, further research was made. Another promising idea was to replaced the Ornstein-Uhelnbeck noise process with a probabilistic noise process. For a further analysis of the noise generation replacement one can review the section [Noise Process](#noise-process). Beside the replacement of the noise process, also a decaying noise rate was introduced, which reduces the noise scaling after each episode. After replacing the noise process, higher rewards were achieved. However, the training still did not lead to solving the environment. In fact many different sets of Hyperparameters were tried (Even long training runs were made with up to 2000 episodes), but none of the configurations lead to solving the environment. Anoter promising avenue was to experiement with scaling the rewards. Wu et al. (2019) found that ReLU performance depends heavily on reward scale. The reason behind this is because if the inputs to ReLUs are too negative (due to large negative rewards propagating through the network), the neuron dies (always outputs zero, which is reffered to as the dying ReLU problem) (Wu et al. 2019). The researcher suggest an adaptive network scaling, however for the sake of simplicity, we experiement with static reward scaling wihtin the step function and passing the scaled reward to the memory. Adding this reward scaling seemed to slighly improve the learning process. However, this still did not lead to solving the environment.

After reviewing the benchmark implementation of Udacity once more, another unexplored avenue appeared to seem promising: Updating the learning less frequent. So far for all made training trials the learning update occured after each time step. This high frequency of learning updates might be too noisy and computational expensive. As the reward mostly stayed below a fixed average reward no matter what hyperparameters were selected, the agent did not seem to learn well potentailly due to too much noise. Also training was taken a lot of time. Thus, to reduce the noise and improve computational efficiency, less frequent learning updates were introduced. In addition, mutliple gradient updates were introduced to ensure efficent and stable training. After the implementation of less frequent learning updates and multiple gradient updates the training of the agent showed signficant improvement and the target score was with the right set of hyperparameters achieved.

## Model Architecture Actor-Critic Network

The Actor-Critic Network, which we already implemented earlier in this course, was utilized as model for solving the reacher problem.

In the script `model.py` one can find the implementation of the Actor and Critic Network. The following code snippet shows the Model architecture. The implementation of the Actor Critic network was taken from the previous exercises of the pendulum from the Udacity course (https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py). Additionaly, for a faster and more stable training, batch normalization layers were added to both the actor and critic network. Ioffe & Szegedy introduced the concept of batch normalization (see https://arxiv.org/pdf/1502.03167) to make the training process more efficent. Also Lillicrap et al. (2015) suggest to utilize batch normalization layers for the actor critic network. 

The Actor (Policy) Model consists of three linear layers, ReLU activation functions, and batch normalization layers. The forward pass computes the action (or policy) for a given state. It uses two fully connected layers, (optionally) with batch normalization and ReLU activations, to process the state input and generate the action. The final action is transformed using the tanh function to ensure it's within a the range [-1, 1]. For the final agent 128 units for the first and 256 units for the second fully connected layer were selected.

The Critic (Value) Model consists of three linear layer, a batch nomarlization layer and ReLU activation functions. The forward pass processes the input state and action to output a Q-value. The state is passed through a fully connected layer, and if batch normalization is enabled, normalization is applied. The state and action are then concatenated and passed through another fully connected layer before producing the final Q-value. For the final agent 128 units for the first and 256 units for the second fully connected layer were selected. 

```bash
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=256, batch_norm=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units) 
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units) 
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.batch_norm = batch_norm
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.batch_norm:
            x = self.bn0(state)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=256, batch_norm = True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.b1 = nn.BatchNorm1d(fcs1_units)
        self.batch_norm = batch_norm
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.batch_norm == True:
            xs = F.relu(self.b1(self.fcs1(state)))
        else:
            xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
```

## Noise Process

After multiple trials with various hyperparameters and the Ornstein Uhlenbeck process for noise creation, somehow the agent did not learn well and seem to get stuck with wrong policies possibly due to too little exploration. Thus, research was made to identify potential alternatives to the Ornstein-Uhlenbeck process for noise generation. Mathereon, Perrin & Sigaud (2019) demonstrated for many different seeds for a small 1D toy example that utilizing the Ornstein-Uhlenbeck process can lead potentially to failures, which might come from an exploration problem. To remove this source of failure, they replace the Ornstein-Uhlenbeck noise process with a "probabilistic noise" (Mathereon, Perrin & Sigaud, 2019). Thus, we also implemented an alternative act function which utilizes a probalistic instead of the Ornstein Uhlenbeck noise.

```python
# Snippet from the ddpg_agent class
def act_probabilistic(self, state, add_noise=True):
    """Returns actions for given state using a stochastic policy with Gaussian noise."""
    state = torch.from_numpy(state).float().to(device)
    self.actor_local.eval()
    with torch.no_grad():
        action = self.actor_local(state)
    self.actor_local.train()
    if add_noise:
        action += torch.randn_like(action) * self.noise_scale
    return np.clip(action.cpu().data.numpy(), -1, 1)
```

This in fact resulted in our case for the agent to learn faster by achieving better rewards earlier in the training process. The following two plots visualize training trials for 40 episodes, which clearly shows higher rewards for the training with probabilistic noise compared to Orhnstein-Uhlenbeck Noise. While training with Ornstein-Uhlenbeck process for noise creation mostly resulted in rewards below 1, the probabilistic noise resulted in way better rewards early on.

![Probabilistic Noise](/results/training_scores/simple_training_plot_trial_01_ProbNoise.png)

*Figure 2: Training for 40 episodes with probabilistic noise*

![Ornstein-Uhlenbeck Noise](/results/training_scores/simple_training_plot_trial_02_UO.png)

*Figure 3: Training for 40 episodes with Ornstein-Uhlenbeck noise*

## Chosen Hyperparameters & brief Explanation of Hyperparameters

Many different Hyperparameters were tested after the final DDGP agent was implemented. The most impactful parameters, which changed the performance of the agent drastically, were the learning frequence, the gradient updates and the units of the fully connected layers. 

The following set of Hyperparameters resulted in solving the environment.

| Hyperparameter         | Value             | Description |
|:------------------------|:------------------|:------------|
| TRIAL_NAME              | Final_Agent        | Name for saving checkpoints, plots, config files. |
| N_EPISODES              | 2000               | Number of episodes to train the agent. |
| MAX_T                   | 2000               | Max time steps per episode. |
| BUFFER_SIZE             | 100000             | Replay buffer size. Improves sample diversity. |
| BATCH_SIZE              | 128                | Number of experiences per training update. |
| GAMMA                   | 0.99               | Discount factor. Higher values emphasize future rewards. |
| TAU                     | 0.001              | Soft update factor for target networks. Smaller results in slower, more stable updates. |
| LEARN_FREQ              | 22                 | Defines after how many steps the agent should learn. |
| GRADIENT_UPDATES        | 10                 | Number of optimization steps per learning trigger. |
| REWARD_SCALING          | false              | Decision whether rewards are scaled. |
| SCALE_FACTOR_REWARD     | 0.01               | Scaling factor applied to rewards (if enabled). |
| **Actor Parameters**    |                    |  |
| LR_ACTOR                | 0.0001             | Learning rate for the actor network. |
| FC1_UNITS (Actor)       | 32                | Units in the first fully connected actor layer. |
| FC2_UNITS (Actor)       | 64                | Units in the second fully connected actor layer. |
| BATCH_NORMALIZATION (Actor) | true           | Whether BatchNorm Layer is applied in actor network. |
| **Critic Parameters**   |                    |  |
| LR_CRITIC               | 0.001              | Learning rate for the critic network. |
| FC1_UNITS (Critic)      | 64                | Units in the first fully connected critic layer. |
| FC2_UNITS (Critic)      | 128                | Units in the second fully connected critic layer. |
| BATCH_NORMALIZATION (Critic) | true          | Whether BatchNorm Layer is applied in critic network. |
| WEIGHT_DECAY (Critic)   | 0.0                | L2 regularization strength for critic network. |
| **Noise Parameters**    |                    |  |
| PROB_NOISE_OR_OU        | 'prob'             | Choice between probabilistic ('prob') or OU noise ('ou'). |
| ACT_NOISE_DECAY         | false              | Decision whether noise decays should be included. |
| **OU Noise Config - Was not used for final Agent**     |                    |  |
| MU                     | 0.0                 | Mean of OU process. |
| THETA                  | 0.15                | Mean reversion rate in OU process. |
| SIGMA                  | 0.15                | Volatility (noise level) in OU process. |
| **Probabilistic Noise Config - Was not used for final Agent** |             |  |
| NOISE_INIT             | 0.99                | Initial noise scale. |
| NOISE_DECAY            | 0.95                | Decay rate of probabilistic noise. |
| NOISE_MIN              | 0.01                | Minimum allowed noise level. |


# Plot of Rewards

The fastest agent solved the environment after after 250 Episodes. *Figure 4* shows the rolling average of the reward over the recent 100 episodes and the episode count, when the environment was considered solved. The target score was set to a bit higher to 32 to better visualize the training scores and ther surpassing of the target. That's why the console printed that the environment was solved after 260 episodes. With a relatively simple actor network with 32 FC1 Units and 64 FC2 Units, and a slightly more complex critic network with 64 FC1 Units and 128 FC2 Units the agent solved the environment the fastests in comparison to the first two succesful training trials. In *Figure 5* one can observe the Reward Score for each episode, the rolling average of the reward over 100 episodes, and the target score. Finally, in *Figure 6* one can see the trained agent absolve the desired task of continously reaching the ball by applying force accordingly on the two joints of the reacher arm.

![Trainings Console of Final Agent](/results/training_scores/TrainingConsole_Agent_03.png)

*Figure 4: Output on the console during and after training of the agent*

![Trainings Process of Final Agent](/results/training_scores/training_scores_trial_Final_Agent_03.png)

*Figure 5: Training scores of the final agent*

![Visualization of Final Agent](/results/trained_agent/Reacher_Version_1_03.gif)

*Figure 6: Visualization of the final agent*

Before the fastest Agent, two more successful Training runs were made. The first agent (see *Figure 7*, *Figure 8*, and  *Figure 9*) had a more complex actor critic network due to the increased units of the first and second layers (FC1 Units = 128, FC2 Units = 256). In addition, the Gradient Updates were set to 4. Other than that the Hyperparameters were left the same compared to the final agent. In *Figure 8* one can observe the Reward Score for each episode, the rolling average of the reward over 100 episodes, and the target score. Finally, in *Figure 9* one can see the trained agent absolve the desired task of continously reaching the ball by applying force accordingly on the two joints of the reacher arm.

![Trainings Process of first agent](/results/training_scores/training_scores_trial_Final_Agent_01.png)

*Figure 7: Output on the console during and after successful training of the first agent*

![Trainings Consoleof first agent](/results/training_scores/TrainingConsole_Agent_01.png)

*Figure 8: Training scores of the first successful agent*

![Trainings Processof first agent](/results/trained_agent/Reacher_Version_1_01.gif)

*Figure 9: Visualization of the first successful agent*

The second successufl agent took even longer to solve the environemnt. After 444 episodes, the agent solved the environment. The adjustment made were only to the Gradient updates. They were increased from 4 to 10. The actor-critic network had the same number of units then the first successful agent. This increase led to a longer training process.  In *Figure 11* one can observe the Reward Score for each episode, the rolling average of the reward over 100 episodes, and the target score. For this agent one could see in *Figure 12* that the agent sometimes even loses the ball but then fixes its mistake and adjust the joints accordingly to reach the ball again.

![Trainings Process of first agent](/results/training_scores/training_scores_trial_Final_Agent_02.png)

*Figure 10: Output on the console during and after successful training of the first agent*

![Trainings Consoleof first agent](/results/training_scores/TrainingConsole_Agent_02.png)

*Figure 11: Training scores of the second successful agent*

![Trainings Processof first agent](/results/trained_agent/Reacher_Version_1_02.gif)

*Figure 12: Visualization of the second successful agent*

# Ideas for Future Work

#TODO ADD FUTURE WORK IDEAS

# References
1. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating deep network training by reducing internal covariate shift. doi:10.48550/ARXIV.1502.03167 (https://arxiv.org/pdf/1502.03167)
2. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., … Wierstra, D. (2015). Continuous control with deep reinforcement learning. doi:10.48550/ARXIV.1509.02971 (https://arxiv.org/pdf/1509.02971)
3. Matheron, G., Perrin, N., & Sigaud, O. (2019). The problem with DDPG: understanding failures in deterministic environments with sparse rewards. Retrieved from http://arxiv.org/abs/1911.11679
4. Pascanu, R., Mikolov, T., & Bengio, Y. (2012). On the difficulty of training Recurrent Neural Networks. doi:10.48550/ARXIV.1211.5063 (https://arxiv.org/pdf/1211.5063)
5. Wu, Y.-H., Sun, F.-Y., Chang, Y.-Y., & Lin, S.-D. (2018). ANS: Adaptive Network Scaling for deep rectifier reinforcement learning models. doi:10.48550/ARXIV.1809.02112 (https://arxiv.org/abs/1809.02112)
6. Zhou, C., Huang, B., & Fränti, P. (2022). A review of motion planning algorithms for intelligent robots. Journal of Intelligent Manufacturing, 33(2), 387–424. doi:10.1007/s10845-021-01867-z (https://doi.org/10.1007/s10845-021-01867-z)