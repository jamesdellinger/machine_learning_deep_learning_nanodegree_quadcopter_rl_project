###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def get_best_episode(results):
    """
    Return the number of the episode that had the highest reward.
    
    Parameters:
        results: Pandas dataframe
    """
    # Number of episodes in the task
    episode_rewards = results.groupby(['episode'])[['reward']].sum()
    episode_of_best_reward = episode_rewards.idxmax()[0]
    return episode_of_best_reward
    

def plot_rewards(results, zoomed_x_range, zoomed_y_range, agent_name, task_name, n):
    """
    Plot the rewards of an agent earned from a task.

    Parameters:
        rewards_list: Pandas dataframe
        zoomed_x_range: tuple delineating the x range visible in the zoomed-in rewards graph
        zoomed_y_range: tuple delineating the y range visible in the zoomed-in rewards graph
        agent_name: string, the name of the agent -- for proper labeling of graphs
        task_name: string, the name of the task -- for proper labeling of graphs
        n: int over which the running mean will be calculated
    """

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize = (18, 6))

    # Set the title
    fig.suptitle("Reward earned by the {} agent on the task: {}".format(agent_name, task_name), fontsize = 18, y=1.05)

    # Total rewards for each episode
    episode_rewards = results.groupby(['episode'])[['reward']].sum()
    
    # In the left graph, plot the total reward for each episode and the running mean.
    smoothed_rewards = episode_rewards.rolling(n).mean() # running_mean of n
    ax[0].plot(smoothed_rewards, label='Running Average Reward (n={})'.format(n))
    ax[0].plot(episode_rewards, color='grey', alpha=0.3, label='Total Reward in Episode')
    ax[0].set_title('{}: \nTotal Reward per Episode in {}'.format(agent_name, task_name))
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Reward')
    ax[0].legend()

    # In the right graph, plot the rewards of an agent from
    # a portion of the episodes of the sample task.
    ax[1].plot(smoothed_rewards, label='Running Average Reward (n={})'.format(n))
    ax[1].plot(episode_rewards, color='grey', alpha=0.3, label='Total Reward in Episode')
    ax[1].set_xlim(zoomed_x_range[0], zoomed_x_range[1])
    ax[1].set_ylim(zoomed_y_range[0], zoomed_y_range[1])
    number_episodes_in_zoomed_range = zoomed_x_range[1] - zoomed_x_range[0]
    ax[1].set_title('{}: \nTotal Reward per Episode in Final {} Episodes of {}'.format(agent_name, number_episodes_in_zoomed_range, task_name))
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Reward')
    ax[1].legend()

    # display the figure
    plt.show()


def getAvgStatByTimestep(results, stat_label, number_of_timesteps):
    """
       Returns a list of average values of the quadcopter for a particular
       statistic (eg. x position) across all episodes, in order of timestep
       from earliest to latest.

        Params
        ======
            results: Pandas dataframe
            stat_label: A string that will be the label of a stat (eg. 'x' or 'y' or 'z')
                                 saved in the results dictionary.
            number_of_timesteps: An int that represents the average duration of all episodes,
                                 in timesteps, for the task
    """

    # A list containing lists. The invidudual lists are ordered according to timestep.
    # Each individual list contains all values (of a particular stat) for the copter at a
    # particular timestep, across all episodes.
    stat_lists_by_step = []

    # Cycle through the results dictionary. Order of episodes is not important.
    # We only ensure that individual stat values themselves are added in proper order
    # according to each successive timestep.
    number_of_episodes = results['episode'].max()
    for i in range(1, number_of_episodes+1):
        # For each episode we will look at all timesteps, if the number of
        # timesteps is less than the default length in timesteps (this method's 
        # number_of_timesteps parameter). Otherwise, we will only look at the 
        # first n timesteps where n is the average length of all episodes in 
        # the task's results.
        results_for_episode = results.loc[results['episode'] == i]
        number_timesteps_in_episode = results_for_episode['time'].count()
        length_of_episode_in_timesteps = min(number_of_timesteps, number_timesteps_in_episode)
        for j in range(length_of_episode_in_timesteps):
            # If no stats have been saved for a particular timestep yet:
            if j > len(stat_lists_by_step) - 1:
                stat_lists_by_step.append([results_for_episode[stat_label].iloc[j]])
            # Else, if there are already stats saved for a given timestep:
            else:
                stat_lists_by_step[j].append(results_for_episode[stat_label].iloc[j])

    # Now compute the average of each list of individual stat values,
    # so that we can have a list that contains the average stat value of the quadcopter
    # at each timestep; this will still be in order of timestep, from earliest to latest.
    average_stat_by_step = []
    for item in stat_lists_by_step:
        average_stat_by_step.append(np.mean(item))

    # Return the list of the copter's average location, across all episodes, at each timestep.
    return average_stat_by_step


def plot_behavior(results, agent_name, task_name):
    """
       Plots six graphs that illustrate the quadcopter's physical behavior during
       a task.

       First graph is quadcopter's average position in, x, y, and z values, across all
       episodes at each timestep up to and including the timestep that represents the
       average episode duration.

       Second graph is the x, y, and z values at each timestep, for the episode when the
       copter had its highest total reward. This illustrates the behavior during the
       copter's best performance.

       The final four graphs plot x, y, z values of the copter at each timestep,
       for 4 episodes chosen at random from the final 100 episodes of the simulation.

        Params
        ======
            results: Pandas dataframe
            best_episode_number: An int that represents the episode number where highest
                                 reward was earned.
            agent_name: string, the name of the agent -- for proper labeling of graphs
            task_name: string, the name of the task -- for proper labeling of graphs
    """

    # Create figure
    fig, ax = plt.subplots(3, 2, figsize = (18, 18))

    # Set the title
    fig.suptitle("Flight behavior of the {} agent on the {} task.".format(agent_name, task_name), y=1.05, fontsize = 18)

    # Number of episodes in the task
    number_of_episodes = results['episode'].max()
    
    # Determine the average episode duration (in terms of number of timesteps) across all
    # episodes run during the task.
    list_of_episode_lengths = []
    for i in range(1, number_of_episodes+1):
        results_for_episode = results.loc[results['episode'] == i]
        number_timesteps_in_episode = results_for_episode['time'].count()
        list_of_episode_lengths.append(number_timesteps_in_episode)
    average_episode_duration = int(np.mean(list_of_episode_lengths))

    # Use the above helper function to get the average x,y,z positions of the quadcopter across
    # all episodes for each timestep.
    average_x_at_each_timestep = getAvgStatByTimestep(results, 'x', average_episode_duration)
    average_y_at_each_timestep = getAvgStatByTimestep(results, 'y', average_episode_duration)
    average_z_at_each_timestep = getAvgStatByTimestep(results, 'z', average_episode_duration)

    # Plot Quadcopter's average position at each timestep, across all episodes that have been run.
    # Display for each timestep up until the average max number of timesteps
    # of all episodes. A sparse number of episodes have far more timesteps than is typical for
    # most episodes. There is no need to show average position at these rarer, higher-numbered timesteps.
    # Locate this plot in upper left hand corner of figure.
    ax[0,0].plot(range(1, len(average_x_at_each_timestep)+1), average_x_at_each_timestep, label='x')
    ax[0,0].plot(range(1, len(average_x_at_each_timestep)+1), average_y_at_each_timestep, label='y')
    ax[0,0].plot(range(1, len(average_x_at_each_timestep)+1), average_z_at_each_timestep, label='z')
    ax[0,0].legend()
    _ = plt.ylim()
    ax[0,0].set_title('Average Location of Quadcopter at each timestep (across all {} episodes)'.format(number_of_episodes))
    ax[0,0].set_xlabel('Timestep (up to n timesteps, where n is avg. episode duration)')
    ax[0,0].set_ylabel('Average Position (in meters)')

    # Next, in the upper right hand corner of the figure, plot
    # the x, y, and z values at each timestep, for the episode when the
    # copter had its highest total reward. We want to see what the
    # copter's behavior was when it was performing at its best.
    best_episode = get_best_episode(results)
    best_episode_results = results.loc[results['episode'] == best_episode]
    ax[0,1].plot((range(1, best_episode_results['time'].count() + 1)), best_episode_results['x'], label='x')
    ax[0,1].plot((range(1, best_episode_results['time'].count() + 1)), best_episode_results['y'], label='y')
    ax[0,1].plot((range(1, best_episode_results['time'].count() + 1)), best_episode_results['z'], label='z')
    ax[0,1].legend()
    _ = plt.ylim()
    ax[0,1].set_title('Location of Quadcopter at each timestep during episode of highest reward (episode {})'.format(best_episode))
    ax[0,1].set_xlabel('Timestep')
    ax[0,1].set_ylabel('Position (in meters)')

    # Finally plot x, y, z values of the copter at each timestep, for 4 episodes chosen at
    # random from the final 100 episodes of the simulation. Display in remaining 4
    # subplots in the figure.
    subplot_indices = [(1,0), (1,1), (2,0), (2,1)]
    # Choose 4 episodes at random from the simulation's final 100 episodes
    random_episodes = np.random.choice(range(number_of_episodes - 100, number_of_episodes + 1), size=4, replace=False)
    # Make a plot for each of the 4 episodes
    for i in range(len(random_episodes)):
        subplot_index_for_episode = subplot_indices[i]
        row = subplot_index_for_episode[0]
        column = subplot_index_for_episode[1]
        episode = random_episodes[i]
        episode_results = results.loc[results['episode'] == episode]
        ax[row, column].plot((range(1, episode_results['time'].count() + 1)), episode_results['x'], label='x')
        ax[row, column].plot((range(1, episode_results['time'].count() + 1)), episode_results['y'], label='y')
        ax[row, column].plot((range(1, episode_results['time'].count() + 1)), episode_results['z'], label='z')
        ax[row, column].legend()
        _ = plt.ylim()
        ax[row, column].set_title('Location of Quadcopter at each timestep in episode {})'.format(episode))
        ax[row, column].set_xlabel('Timestep')
        ax[row, column].set_ylabel('Position (in meters)')

    # Display the figure
    plt.tight_layout()
    plt.show()
