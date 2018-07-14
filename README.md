# Project: Deep Reinforcement Learning Quadcopter Controller
*Teach a Quadcopter how to fly.*
### For Udacity's Machine Learning Engineer and Deep Learning Nanodegrees
<img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/mlndlogo.png" height="140">     <img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/dlndlogo.png" height="140">

### Topic: Deep Reinforcement Learning

### Overview:

* I designed a reinforcement learning task for flying a quadcopter in a simulated environment, and built an agent that autonomously learned to perform the task.

    <img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/images/parrot-ar-drone.jpg" height="200">
* The goal of the task was for an agent to learn to take off, ascend to a fixed height, and keep the quadcopter hovering at this height while remaining directly above the position on the ground from where it took off.
* This involved my crafting a straightforward reward function that would incentivize an agent to first take off, and then remain hovering at a constant altitude.
* My agent uses the [deep deterministic policy gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) invented by Lillicrap, Timothy P., et al.
* I first implemented a DDPG agent that could solve OpenAI Gym's [MountainCarContinuous-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0) task.

    <img src="https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/images/MountainCarContinuous-v0.png" height="200">
* I then tweaked this DDPG model's hyperparameters as well as the hidden layer architecture of its actor and critic networks in order to achieve better performance in the quadcopter's simulated physics environment.

### Concepts:

* Deep Q Learning
* Reinforcement learning in continuous spaces
* Actor/Critic models
* DDPG
* Ornstein-Uhlenbeck noise
*

### My completed project:

* [ipython notebook](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_Project.ipynb) / [html version](http://htmlpreview.github.com/?https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/report.html) / [pdf version](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_Project.pdf)

### Project Grading and Evaluation:

* [Project Review](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_project_review.pdf)

* [Project Grading Rubric](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/Quadcopter_Project_grading_rubric.pdf)

## Project Overview
The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. This is what you are going to do in this project!

## Project Highlights
In this project, you will design your own reinforcement learning task and an agent to complete it. Note that getting a reinforcement learning agent to learn what you actually want it to learn can be hard, and very time consuming. For this project, we strongly encourage you to take the time to tweak your task and agent until your agent is able to demonstrate that it has learned your chosen task, but this is not necessary to complete the project. As long as you take the time to describe many attempts at specifying a reasonable reward function and a well-designed agent with well-informed hyperparameters, this is enough to pass the project.

### Dependencies:

* [requirements.txt](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/requirements.txt)

* [Anaconda .yml file](https://github.com/jamesdellinger/machine_learning_deep_learning_nanodegree_Quadcopter_RL_project/blob/master/quadcopter_project.yml)
