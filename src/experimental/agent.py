"""
Implements a Deep Q-Network (DQN) agent for reinforcement learning.

This agent learns a policy to select actions in an environment to maximize
cumulative reward. It uses a replay buffer to store experiences and learn
off-policy, and employs separate online and target networks with periodic
updates for stability. Epsilon-greedy exploration is used during training.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input # Removed Concatenate as branching NN is experimental
from collections import deque
import random
import os


class ReplayBuffer:
    """Stores experiences and provides random batches for training."""
    def __init__(self, buffer_size: int):
        """Initializes the replay buffer."""
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        """Adds an experience tuple (s, a, r, s', d) to the replay buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a random mini-batch of experiences from the buffer.
        Returns None if the buffer doesn't have enough samples.
        """
        if len(self.buffer) < batch_size:
            return None # Not enough samples yet
        batch = random.sample(self.buffer, batch_size)

        # Unpack batch into separate numpy arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32) # Ensure correct dtype
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        # Dones should be float for multiplication with gamma
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning Agent for controlling a single entity (e.g., traffic signal).
    """
    def __init__(self, state_size: int, action_size: int, agent_id: str, config: dict):
        """
        Initializes the DQN Agent.

        Args:
            state_size (int): Dimension of the input state vector.
            action_size (int): Number of possible discrete actions.
            agent_id (str): Unique identifier for this agent.
            config (dict): Dictionary of agent hyperparameters. Expected keys:
                           learning_rate, gamma, epsilon_start, epsilon_min,
                           epsilon_decay_steps, replay_buffer_size, batch_size,
                           target_update_frequency, nn_layers.
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters from config dictionary
        self.learning_rate = config.get("learning_rate", 0.001)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay_steps = config.get("epsilon_decay_steps", 100000)
        # Calculate linear decay rate
        self.epsilon_decay_rate = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps \
                                  if self.epsilon_decay_steps > 0 else 0
        self.buffer_size = config.get("replay_buffer_size", 50000)
        self.batch_size = config.get("batch_size", 64)
        # Update target network based on number of learning steps (calls to learn())
        self.target_update_frequency = config.get("target_update_frequency", 1000)
        self.nn_layers = config.get("nn_layers", [128, 128]) # Default hidden layers

        # Build Networks
        self.online_network = self._build_q_network(self.state_size, self.action_size, self.nn_layers)
        self.target_network = self._build_q_network(self.state_size, self.action_size, self.nn_layers)
        self.update_target_network() # Initialize target network weights

        # Optimizer and Loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Using Huber loss can sometimes be more robust to outliers than MSE
        self.loss_fn = tf.keras.losses.Huber()
        # self.loss_fn = tf.keras.losses.MeanSquaredError() # Alternative

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Counters
        self.learn_step_counter = 0 # Tracks calls to learn() for target network updates

    def _build_q_network(self, input_shape, output_shape, hidden_layers):
        """Builds a simple sequential Q-Network."""
        model = keras.Sequential(name=f"QNetwork_{self.agent_id}")
        model.add(Input(shape=(input_shape,)))
        for layer_size in hidden_layers:
            model.add(Dense(layer_size, activation='relu'))
        # Output layer for Q-values (linear activation)
        model.add(Dense(output_shape, activation='linear'))
        # No compile needed here; training handled manually with GradientTape
        print(f"Agent {self.agent_id}: Built Q-Network with layers {hidden_layers}, output {output_shape}")
        model.summary() # Print model summary
        return model

    def select_action(self, state_tensor):
        """
        Selects an action using the epsilon-greedy policy based on the online network.

        Args:
            state_tensor (tf.Tensor): The current state observation as a TensorFlow tensor
                                     with a batch dimension (shape [1, state_size]).

        Returns:
            int: The chosen action index.
        """
        if np.random.rand() < self.epsilon:
            # Explore: Choose a random action
            action = np.random.randint(0, self.action_size)
        else:
            # Exploit: Choose the action with the highest predicted Q-value
            q_values = self.online_network(state_tensor, training=False) # Shape [1, action_size]
            action = tf.argmax(q_values[0]).numpy() # Get index of max Q-value

        # Decay epsilon (linear decay) after action selection
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_rate
        # Ensure epsilon doesn't go below min
        self.epsilon = max(self.epsilon_min, self.epsilon)

        return action

    def select_greedy_action(self, state_tensor):
        """Selects the greedy action (highest Q-value) without exploration."""
        q_values = self.online_network(state_tensor, training=False)
        action = tf.argmax(q_values[0]).numpy()
        return action

    def store_experience(self, experience):
        """Stores an experience tuple (s, a, r, s', d) in the replay buffer."""
        # Experience expected as (NumPy array, int, float, NumPy array, bool)
        self.replay_buffer.add(experience)

    # Use tf.function for potential performance optimization of the training step
    @tf.function
    def _perform_gradient_descent(self, states, actions, rewards, next_states, dones):
        """Performs the gradient descent step using GradientTape."""

        # Calculate target Q-values using the target network
        # Q_target = r + gamma * max_a'( Q_target(s', a') ) * (1 - done)
        max_next_q_values = tf.reduce_max(self.target_network(next_states, training=False), axis=1)
        target_q = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        with tf.GradientTape() as tape:
            # Get predicted Q-values for the actions taken using the online network
            all_predicted_q = self.online_network(states, training=True) # Shape [batch_size, action_size]
            # Select Q-values corresponding to the actions actually taken in the batch
            # Indices for gather_nd: [[0, action0], [1, action1], ...]
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            predicted_q_for_taken_actions = tf.gather_nd(all_predicted_q, action_indices)

            # Calculate loss between target Q and predicted Q for taken actions
            loss = self.loss_fn(target_q, predicted_q_for_taken_actions)

        # Calculate gradients and apply them using the optimizer
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))

        return loss

    def learn(self):
        """
        Performs a learning update if enough samples are in the replay buffer.
        Samples a batch, calculates loss, performs gradient descent, and updates
        the target network periodically.

        Returns:
            float or None: The calculated loss for this learning step, or None if
                           learning did not occur (e.g., buffer too small).
        """
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None # Not enough samples yet

        states_np, actions_np, rewards_np, next_states_np, dones_np = batch

        # Convert numpy arrays to TensorFlow tensors
        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions_np, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states_np, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones_np, dtype=tf.float32)

        # Perform the gradient descent step
        loss = self._perform_gradient_descent(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf)

        self.learn_step_counter += 1

        # Periodically update the target network
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.update_target_network()
            # print(f"Agent {self.agent_id}: Target network updated at learn step {self.learn_step_counter}")

        return loss

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_network.set_weights(self.online_network.get_weights())

    def save_model(self, filepath):
        """Saves the weights of the online network."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.online_network.save_weights(filepath)
            print(f"Agent {self.agent_id}: Model weights saved to {filepath}")
        except Exception as e:
            print(f"Agent {self.agent_id}: Error saving model weights to {filepath}: {e}")

    def load_model(self, filepath):
        """Loads weights into the online network (and updates target network)."""
        if not os.path.exists(filepath):
             print(f"Agent {self.agent_id}: Error loading model weights. File not found: {filepath}")
             return False
        try:
            self.online_network.load_weights(filepath)
            self.update_target_network() # Ensure target network matches loaded weights
            print(f"Agent {self.agent_id}: Model weights loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Agent {self.agent_id}: Error loading model weights from {filepath}: {e}")
            return False

    def get_epsilon(self):
        """Returns the current epsilon value."""
        return self.epsilon