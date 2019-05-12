# Double Joint Arm Placement using DDPG

## Introduction

In this project we will train an agent with double jointed arm to move to its target (in this case green spheres) and be in contact with its target.
The environment and a trained agent looks like this:

![Alt Text](images/reacher.gif)


The simulated environment is Unity based Reacher environment wherein the agent has an arm having two joints and the aim is to move the arm in contact to its target and keep it connected to target for as long as possible in an episode.
**A reward of +0.1 is received on each time step that the agent is at its target**. Thus, the goal of agent is to maintain its position at target for as many time steps as possible.

The simulated environment provides a simplified **state space having 33 dimensions** corresponding to the agent's position, rotation, velocity, and angular velocities of the arm. **There are four actions in continuous domain forming a vector with four numbers each representing torque applicable to two joints :**


We will assume the environment solved when the agent is able to achieve more than or equal to 30 score on average over 100 episodes.

## Installation Instructions

1. Download the appropriate Unity environment according to the operating system and decompress the contents into your working directory:
    - Linux: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX : [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit) : [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit) : [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    You can use the above downloaded environment to view how a trained agent behaves in the given environment setup.

2. Since the project is developed using Python and Pytorch some necessary packages need to be installed. Install the necessary libraries and packages mentioned in **requirements.txt**. To install the necessary packages either use pip or create a new conda environment and install the minimum required packages as mentioned in the requirements file. To set up a python environment to run code using conda, you may follow the instructions below:

    Create and activate a new environment with Python 3.6 and install dependencies
    
    - Linux or Mac:
      ```
      conda create --name env-name python=3.6
      source activate env-name
      ```
    
    - Windows:
      ```
      conda create --name env-name python=3.6
      activate env-name
      ```
  
    - Then install the dependecies using 
      ```
      pip install -r requirements.txt
      ```
3. To get more information about unity environments, follow the instructions in this [link](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

4. Run the cells in Jupyter notebook to train a new agent on this environment. Pretrained agent's weights are also present in weights folder.

