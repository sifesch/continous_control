[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Continous Control - Let's Learn to Continously Reach a Moving Target

This project implements a Deep Deterministic Policy Gradient (DDPG) agent with probabilistic noise to control a double-jointed arm for reaching a continously moving target. The agent interacts with the Unity ML-Agents Reacher environment. 

In the upcoming section [Introduction](#introduction), an introduction into the project and the environment follows. Further, one can get familiar with the necessary dependencies to run this project on your own local device in the [Getting Started](#getting-started) section. The previously mentioned sections are mainly authored by Udacity and were copied into the project. Some minor adjustments were made to these sections. In the section [Preview of the trained Agent](#preview-of-the-trained-agent) one can already get a peak in the trained agents in action. After sucessfully downloading and installing the prerequisites, in the section [Instructions](#instructions) one can get to know how to set the training configurations, run the training for the DDPG Agent, and how to watch the trained agent. The Training can be executed via terminal. Finally, one can have a glimpse at the structure of the repository in [Structure of the Repository](#structure-of-the-repository). A detailed report of this implementation can be found in the report markdown `Report.md`.

## Table of Contents

<!-- TOC -->

- [Continous Control - Let's Learn to Continously Reach a Moving Target](#continous-control---lets-learn-to-continously-reach-a-moving-target)
  - [Table of Contents](#table-of-contents)
  - [Project Details](#project-details)
    - [Introduction](#introduction)
    - [Solving the Environment](#solving-the-environment)
  - [Preview of the trained Agent](#preview-of-the-trained-agent)
  - [Getting Started](#getting-started)
    - [Python Dependencies](#python-dependencies)
    - [Getting the Reacher Environment](#getting-the-reacher-environment)
  - [Instructions](#instructions)
    - [Setting the Training Configurations](#setting-the-training-configurations)
    - [Running the Training](#running-the-training)
    - [Watching the trained Agent](#watching-the-trained-agent)
  - [Structure of the Repository](#structure-of-the-repository)

<!-- /TOC -->
<!-- /TOC -->
<!-- /TOC -->

## Project Details

### Introduction

For this project, we work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. 

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Preview of the trained Agent

![Reacher Version 1](results/trained_agent/Reacher_Version_1_03.gif)

## Getting Started

### Python Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6. In case you do not have conda installed, ensure to install anaconda on your system: https://www.anaconda.com/docs/getting-started/anaconda/install 

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/sifesch/continous_control
cd continous_control/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

### Getting the Reacher Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the in the `continuous-control/` folder, and unzip (or decompress) the file. 

## Instructions

First, one has to set the configurations for the general settings, the hyperparameters and the noise. Then one can run the training with the defined configurations. Afterwards one can watch the trained agent. To watch the trained agent, the configurations need to be set to the same ones of the respective trained agent (see therefore `results/configurations/`).

### Setting the Training Configurations

1. Open the `config/train_config.yaml` file.
2. Define the configurations for the Training. View the following yaml snippet to get an understanding what possibilties there are 

```yaml
HYPERPARAMETERS:
  TRIAL_NAME: 'Test'            # Defines name logic for saving (checkpoint weights, training plot, configuration file)
  N_EPISODES: 101               # Defines how many episodes the agent will be trained   
  MAX_T: 2000                   # Defines the maximum number of time steps the agent can take per episode
  BUFFER_SIZE: 100000           # replay buffer size, defines the max number of experiences the buffer to hold
  BATCH_SIZE: 32                # minibatch size, defines the number of experiences sampled from the buffer to train the agent on each learning update
  GAMMA: 0.99                   # discount factor, defines how important future vs. immedieate rewards are. Close to 1 means agent careas a lot about future rewards, zero means the agent only cares about immediate rewards
  TAU: 0.001                    # for soft update of target parameters, defines how much to update the target networks each step, makes learning stable by slowly updating target networks
  LEARN_FREQ: 100               # defines after how many steps the agent should learn, until then the agent acts and collects experiences
  GRADIENT_UPDATES: 1           # Decision how many time the gradients are updated during each step
  REWARD_SCALING: False         # Defines if the reward should be scaled
  SCALE_FACTOR_REWARD: 0.01     # Defines by how much the rewards should be scaled
  ACTOR_PARAMS:                 # Parameters for the Actor Network
    BATCH_NORMILIZATION: True   # Defines wheter to include Batch Normalization Layers in the Network for more stable learning
    LR_ACTOR: 0.0001            # learning rate of the actor 
    FC1_UNITS: 32               # Units of the first linear layer
    FC2_UNITS: 64               # Units of the second linear layer
  CRITIC_PARAMS:                # Parameters for the Critic Network
    BATCH_NORMILIZATION: True   # Defines wheter to include Batch Normalization Layers in the Network for more stable learning
    LR_CRITIC: 0.001            # learning rate of the critic
    FC1_UNITS: 64               # Units of the first linear layer
    FC2_UNITS: 128               # Units of the second linear layer
    WEIGHT_DECAY: 0.0           # L2 weight decay to regularize the critic to prevent overfitting. Makes critic more stable and general by preventing noisy Q-valuse from the replay buffer (usually )

NOISE:                          # Noise Settings
  GENERAL:                      # General Settings for decision of Noise Type and including noise decay
    PROB_NOISE_OR_OU: 'prob'    # choice of 'prob' or 'ou' noise.
    ACT_NOISE_DECAY: FALSE      # Decision to activate a decaying noise rate
  OUNoise_Config:               # Configurations for Ornstein-Uhlenbeck-Prozess
    MU: 0.0                     
    THETA: 0.15
    SIGMA: 0.15                 
  ProbNoise_Config:             # Configurations for Probabilistic Noise
    NOISE_INIT: 0.99            # Initialization of noise factor (If no decay this will be always the noise factor)
    NOISE_DECAY: 0.95           # Decaying rate for probabilistc noise
    NOISE_MIN: 0.01             # Min Value for Decaying Noise Rate
```
### Running the Training

1. Navigate to directory `continous_control` in your command terminal, ensure the dependencies are installed properly and the respective conda environment is activated. Then run the training by running the following command:
 ```bash
 python3 src/main_train.py
 ```

### Watching the trained Agent

1. Open the `src/agent_visualizer.py` file.
2. Define in section `if __name__ == '__main__':` the weights of interests for the actor and critic network, indicating the individual trained agent you want to review. In addition, define the file name of the Banana environment (depending on which environment you are using). Finally, you can decide on how many steps you want to watch the agent.

```python
if __name__ == "__main__":
    visualizer = ReacherAgentVisualizer(file_path = 'Reacher_Linux/Reacher.x86_64', # Set to respective file name
                 actor_path = 'checkpoints/checkpoint_actor_trial_configurations_MedUnitsActCri_LF22_GU_10.pth', # set to respective weights for actor network
                 critic_path = 'checkpoints/checkpoint_critic_trial_configurations_MedUnitsActCri_LF22_GU_10.pth',
                 seed = 2) # set to respective weights for critic network
    visualizer.run(max_t = 200) # Set steps
```

3. Ensure that the configurations in `config/train_config.yaml` are set like the ones of the agent you want to watch. To get the respective configurations of the trained agent one can check the created configurations in `results/configurations/`.
4. Navigate to the directory `continous_control` in your command terminal, ensure the dependencies are installed properly and the respective conda environment is activated. Then run the following command to observe what the trained agent learned:
 ```bash
 python3 src/agent_visualizer.py
 ```

## Structure of the Repository

```
├── config                      # Folder containing the configuration for hyperparameters and noise
│   └── train_config.yaml       # yaml file containg the Hyperparameter and noise configuration
├── python                      # Python setup (Needs to be put in this folder, see Getting Started Section) 
├── Reacher_Linux               # Reacher Environment (Name could vary depending on your OS, needs to be put into this folder, see Getting Started Section)
├── results                     # result folder containing the checkpoint weights, configurations for the run and training scores visualization.
│   ├── checkpoints             # folder containing the weights of the trained actor and critic networks
│   ├── configurations          # folder containing the hyperparameter configurations of the trained agents
│   ├── trained_agent           # folder containing a gif of the trained agent
│   └── training_scores         # folder containing the scores as numpy files and the 
├── src                         # Main files for the Actor Critic Network, the DDPG Agent, the Training
│   ├── agent_visualizer.py     # Script to visualize an agent based on the weights
│   ├── config_loader.py        # script to load the configurations from the config/train_config.yaml file.
│   ├── ddpg_agent.py           # Containing the implementation of the DDPG Agent
│   ├── main_train.py           # main training script
│   ├── model.py                # Containing the Actor and Critic Model Architecture
│   └── utils.py                # Contains helper functions to create plots
├── README.md                   # README you are currently reading
└── Report.md                   # Report about the learning algorithm, reward plots and future ideas for project improvements
```
