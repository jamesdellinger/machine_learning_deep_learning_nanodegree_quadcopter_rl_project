import numpy as np
import gym
from model_ddpg_agent_mountain_car_continuous import Actor, Critic, ReplayBuffer


class DDPG_Agent_Mountain_Car_Continuous():
    """
    Reinforcement Learning agent that learns by using DDPG, or Deep 
    Deterministic Policy Gradients. An actor-critic method, but with 
    the key idea that the underlying policy function used is deterministic 
    in nature, with some noise added in externally to produce the desired 
    stochasticity in actions taken.

    Algorithm originally presented in this paper:

    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep 
    Reinforcement Learning

    https://arxiv.org/pdf/1509.02971.pdf
    
    Code in this class, as well as from the Actor, Critic, and 
    ReplayBuffer classes in model_ddpg_agent_mountain_car_continuous.py was 
    adopted from sample code that introduced DDPG in the Reinforcement Learning 
    lesson in Udacity's Machine Learning Engineer nanodegree.
    
    Certain modifications to the Udacity approach, such as using an 
    initial exploration policy to warm up (3 times longer than typical) a larger memory buffer 
    (batch size of 256 instead of 64) was inspired by another DDPG solution 
    to OpenAI Gym's 'MountainCarContinuous-v0' environment. This 
    implementation can be viewed at: 
    
    https://github.com/lirnli/OpenAI-gym-solutions/blob/master/Continuous_Deep_Deterministic_Policy_Gradient_Net/DDPG%20Class%20ver2.ipynb
    
    Note that we will need two copies of each model - one local and one target. 
    This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, 
    and is used to decouple the parameters being updated from the ones that are 
    producing target values.
    """
    
    def __init__(self, task):
        self.task = task

        # For OpenAI Gym envs, the following attributes need 
        # to be calculated differently from from a standard 
        # Quadcopter task.
        self.action_size = task.action_space.shape[0]
        self.action_low = task.action_space.low[0]
        self.action_high = task.action_space.high[0]
        
        # If task is OpenAi Gym 'MountainCarContinuous-v0' environment
        # Adjust state size to take advantage of action_repeat parameter.
        # Must do this here when running the 'MountainCarContinuous-v0' environment.
        self.action_repeat = 3
        self.state_size = task.observation_space.shape[0] * self.action_repeat

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 256
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001   # for soft update of target parameters

    def reset_episode(self):
        state = self.task.reset()
        
        # Since the task is OpenAi Gym 'MountainCarContinuous-v0' environment, 
        # we must expand the state returned from the gym environment according to 
        # our chosen action_repeat parameter value.
        state = np.concatenate([state] * self.action_repeat) 
        
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size*3:    # Warm up period is 3 times longer than typical
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns action(s) for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return action # OU noise will be added outside the agent's class

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)