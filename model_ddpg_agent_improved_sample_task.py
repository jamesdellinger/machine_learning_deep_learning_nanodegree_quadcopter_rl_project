import random
import numpy as np
import copy
from collections import namedtuple, deque
import keras


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    #def sample(self, batch_size=64):
    def sample(self, batch_size=256):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)
    
    
class OUNoise:
    """
    Ornstein-Uhlenbeck process. Generates random samples from a Gaussian 
    (Normal) distribution, but each sample affects the next one such that 
    two consecutive samples are more likely to be closer together than 
    further apart.
    
    The OU process tends to settle down close to the specified mean over time. 
    When used to generate noise, we can specify a mean of zero, and that will 
    have the effect of reducing exploration as we make progress on 
    learning the task.
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    

class Actor:
    """
    Actor (Policy) Model, using Deep Deterministic Policy Gradients 
    or DDPG. An actor-critic method, but with the key idea that the 
    underlying policy function used is deterministic in nature, with 
    some noise added in externally to produce the desired stochasticity 
    in actions taken.

    Algorithm originally presented in this paper:

    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep 
    Reinforcement Learning

    https://arxiv.org/pdf/1509.02971.pdf

    """

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = keras.layers.Input(shape=(self.state_size,), name='states')
        
        # Kernel initializer with fan-in mode and scale of 1.0
        kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        # Kernel L2 loss regularizer with penalization param of 0.01
        # kernel_regularizer = keras.regularizers.l2(0.01)

        # Add hidden layers
        net = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(states)
        net = keras.layers.Dense(units=300, activation='elu', kernel_initializer=kernel_initializer)(net)
        #net = keras.layers.Dense(units=32, activation='elu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(net)

        # Kernel initializer for final output layer: initialize final layer weights from 
        # a uniform distribution of [-0.003, 0.003]
        # final_layer_initializer = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
        
        # Add final output layer with sigmoid activation
        raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions', kernel_initializer=kernel_initializer)(net)

        # Note that the raw actions produced by the output layer are in a [0.0, 1.0] range 
        # (using a sigmoid activation function). So, we add another layer that scales each 
        # output to the desired range for each action dimension. This produces a deterministic 
        # action for any given state vector.
        actions = keras.layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = keras.models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        # These gradients will need to be computed using the critic model, and 
        # fed in while training. This is why they are specified as part of the 
        # "inputs" used in the training function.
        action_gradients = keras.layers.Input(shape=(self.action_size,))
        loss = keras.backend.mean(-action_gradients * actions)

        # Define optimizer and training function
        # Use learning rate of 0.0001
        optimizer = keras.optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras.backend.function(
            inputs=[self.model.input, action_gradients, keras.backend.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model, using Deep Deterministic Policy Gradients 
    or DDPG. An actor-critic method, but with the key idea that the 
    underlying policy function used is deterministic in nature, with 
    some noise added in externally to produce the desired stochasticity 
    in actions taken.

    Algorithm originally presented in this paper:

    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep 
    Reinforcement Learning

    https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers. The critic model needs to map (state, action) pairs to 
        # their Q-values. This is reflected in the following input layers.
        states = keras.layers.Input(shape=(self.state_size,), name='states')
        actions = keras.layers.Input(shape=(self.action_size,), name='actions')
        
        # Kernel initializer with fan-in mode and scale of 1.0
        kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        # Kernel L2 loss regularizer with penalization param of 0.01
        kernel_regularizer = keras.regularizers.l2(0.001)

        # Add hidden layer(s) for state pathway
        net_states = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(states)
        # net_states = keras.layers.Dense(units=64, activation='elu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(actions)
        # net_actions = keras.layers.Dense(units=64, activation='elu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(net_actions)

        # Combine state and action pathways. The two layers can first be processed via separate 
        # "pathways" (mini sub-networks), but eventually need to be combined.
        net = keras.layers.Add()([net_states, net_actions])

        # Add more layers to the combined network if needed
        net = keras.layers.Dense(units=300, activation='elu', kernel_initializer=kernel_initializer)(net)

        # Kernel initializer for final output layer: initialize final layer weights from 
        # a uniform distribution of [-0.003, 0.003]
        # final_layer_initializer = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
        
        # Add final output layer to produce action values (Q values). The final output 
        # of this model is the Q-value for any given (state, action) pair.
        Q_values = keras.layers.Dense(units=1, activation=None, name='q_values', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(net)
        # kernel_regularizer=kernel_regularizer
        
        # Create Keras model
        self.model = keras.models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        # Use learning rate of 0.001
        optimizer = keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions). We also need 
        # to compute the gradient of every Q-value with respect to its corresponding action 
        # vector. This is needed for training the actor model. 
        # This step needs to be performed explicitly.
        action_gradients = keras.backend.gradients(Q_values, actions)

        # Finally, a separate function needs to be defined to provide access to these gradients. 
        # Define an additional function to fetch action gradients (to be used by actor model).
        self.get_action_gradients = keras.backend.function(
            inputs=[*self.model.input, keras.backend.learning_phase()],
            outputs=action_gradients)