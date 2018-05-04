# Content: Reinforcement Learning
## Project: Teach a Quadcopter How to Fly

## Project Overview
The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. This is what you are going to do in this project!

## Project Highlights
In this project, you will design your own reinforcement learning task and an agent to complete it. Note that getting a reinforcement learning agent to learn what you actually want it to learn can be hard, and very time consuming. For this project, we strongly encourage you to take the time to tweak your task and agent until your agent is able to demonstrate that it has learned your chosen task, but this is not necessary to complete the project. As long as you take the time to describe many attempts at specifying a reasonable reward function and a well-designed agent with well-informed hyperparameters, this is enough to pass the project.

## Project Instructions
You are encouraged to use the workspace in the next concept to complete the project. Alternatively, you can clone the project from the [GitHub repository](https://github.com/udacity/RL-Quadcopter-2). If you decide to work from the GitHub repository, make sure to edit the provided requirements.txt file to include a complete list of pip packages needed to run your project.

The concepts following the workspace are optional and provide useful suggestions and starter code, in case you would like some additional guidance to complete the project.

## Submitting the Project
If submitting from the workspace, simply click the "Submit Project" button at the bottom of the workspace window.

If submitting the project from your local computer, either submit the GitHub link to your repository, or compress all files in the project folder into a single archive for upload.
- This should include the `Quadcopter_Project.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An **HTML** export of the project notebook with the name **report.html**. This file *must* be present for your project to be evaluated.

If you are having any problems submitting your project or wish to check on the status of your submission, please email us at **machine-support@udacity.com** or visit us in the [discussion forums](http://discussions.udacity.com).

### Evaluation
Your project will be reviewed by a Udacity reviewer against the project rubric. Review this rubric thoroughly, and self-evaluate your project before submission. All criteria found in the [rubric](https://review.udacity.com/#!/rubrics/1189/view) must meet specifications for you to pass.
