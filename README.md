# Project: Deep Reinforcement Learning Quadcopter Controller
*Teach a Quadcopter how to fly.*
### For Udacity's Machine Learning Engineer and Deep Learning Nanodegrees
<img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/mlndlogo.png" height="140">     <img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/dlndlogo.png" height="140">

### Topic: Deep Reinforcement Learning

### Overview:

* I designed a reinforcement learning task for flying a quadcopter in a simulated environment, and built an agent that autonomously learned to perform the task.

    <img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/images/parrot-ar-drone.jpg" height="200">
* My agent had to learn how to best manage the quadcopter's four points of thrust in order to complete the behavior mandated by the learning task.
* The goal of my task was for an agent to learn to take off, ascend to a fixed height, and keep the quadcopter hovering at this height while remaining directly above the position on the ground from where it took off.
* This involved my crafting a straightforward reward function that would incentivize an agent to first take off, and then remain hovering at a constant altitude.
* My agent uses the [deep deterministic policy gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) invented by Lillicrap, Timothy P., et al.
* I first implemented a DDPG agent that could solve OpenAI Gym's [MountainCarContinuous-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0) task.

    <img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/images/MountainCarContinuous-v0.png" height="200">
* I then tweaked this DDPG model's hyperparameters such as its learning rate, soft target update parameter, and discount factor, as well as the hidden layer architectures of its actor and critic networks, in order to achieve better performance in the quadcopter's simulated physics environment.

### Concepts:

* Deep Q Learning
* Reinforcement learning in continuous spaces
* DDPG
* Actor/Critic models
* Off-policy learning with local/target networks
* Soft target updates
* Memory replay buffer
* Initial exploration policy/warm-up phase
* Action repeat
* Batch normalization
* elu, relu, and tanh activation functions
* Ornstein-Uhlenbeck noise

### My completed project:

* [ipython notebook](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_Project.ipynb) / [html version](http://htmlpreview.github.com/?https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/report.html) / [pdf version](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_Project.pdf)

### Project Grading and Evaluation:

* [Project Review](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_project_review.pdf)

* [Project Grading Rubric](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_Project_grading_rubric.pdf)

### Dependencies:

* [requirements.txt](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/requirements.txt)

* [Anaconda .yml file](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/quadcopter_project.yml)
