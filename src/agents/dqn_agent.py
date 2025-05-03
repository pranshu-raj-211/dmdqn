"""Models a single DQN agent making decisions for signals at one intersection.

Based on our problem representation, we want to have multiple agents, one for each intersection
in the network. Each of these agents are fed the information of the intersection they are being
trained on (local) and the information of their immediate neighbors (0,1,2,3,4 depending on
network topology).

In this module, we initialize a single such agent. We define the config as input while instantiating
the agent object, which contains information about the state representation as well as hyperparameters.
More on this in the configs/ dir.

The basic functionality needed for this class is to instantiate the neural networks (target and online),
enable buffer replay to learn from experience, use epsilon greedy methods to select actions, apply
updates to the target network and other utilities like model saving and loading.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers  # type: ignore
from tensorflow.keras.layers import Dense, Input, Concatenate  # type: ignore
from collections import deque
import random
from log_config import logger


EXPLORATION_ONLY_STEPS = 10_000


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience: tuple):
        """Adds an experience tuple (s, a, r, s', d) to the replay buffer."""
        state, action, reward, next_state, done = experience

        logger.debug(
            f"Original state shape: {len(state)},{state} , next_state shape: {len(next_state)}, {next_state}"
        )

        state_copy = np.array(state, copy=True)
        next_state_copy = np.array(next_state, copy=True)
        logger.debug(
            f"State shape before {state_copy.shape}, {next_state_copy.shape}, {action}, {reward}, {done}"
        )

        state_copy = np.squeeze(state_copy, axis=0)
        next_state_copy = np.squeeze(next_state_copy, axis=0)
        logger.debug(
            f"State shape after {state_copy.shape}, {next_state_copy.shape}, {action}, {reward}, {done}"
        )
        if len(state_copy) != len(next_state_copy):
            logger.error(
                f"State and next_state sizes are inconsistent: {len(state_copy)} vs {len(next_state_copy)}"
            )
            return

        self.buffer.append((state_copy, action, reward, next_state_copy, done))
        logger.debug(f"Replay buffer size after adding: {len(self.buffer)}")

    def sample(self, batch_size: int):
        """Samples a random mini-batch of experiences from the buffer."""
        if len(self.buffer) < batch_size:
            return None  # Not enough samples yet
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        epsilon = 1e-8
        normalized_rewards = (rewards - reward_mean) / (reward_std + epsilon)

        # logger.debug("Debugging ReplayBuffer.sample:")
        # logger.debug(f"Batch size: {batch_size}")
        # logger.debug(f"Number of experiences in buffer: {len(self.buffer)}")
        # logger.debug(f"Sampled states:{states.shape}\n {states}")
        # logger.debug(f"Sampled actions:{actions.shape}\n {actions}")
        # logger.debug(f"Sampled rewards:{rewards.shape}\n {rewards}")
        # logger.debug(f"Sampled next_states:{next_states.shape}\n {next_states}")
        # logger.debug(f"Sampled dones:{dones.shape}\n {dones}")

        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(normalized_rewards, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
        return states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning Agent for traffic signal control at a single intersection.
    """

    def __init__(self, state_size: int, action_size: int, agent_id: str, config: dict):
        """
        Initializes the DQN Agent.

        Args:
            state_size (int): Dimension of the input state vector (e.g., 89).
            action_size (int): Number of possible discrete actions (4).
            agent_id (str): Unique identifier for this agent (junction ID).
            config (dict): Dictionary of agent hyperparameters.
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 0.001)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay_steps = config.get("epsilon_decay_steps", 100000)
        self.epsilon_decay_rate = (
            (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps
            if self.epsilon_decay_steps > 0
            else 0
        )  # Linear decay
        self.buffer_size = config.get("replay_buffer_size", 10000)
        self.batch_size = config.get("batch_size", 128)
        self.target_update_frequency = config.get(
            "target_update_frequency", 1000
        )  # number of learning steps
        self.nn_layers = config.get("nn_layers", [64, 64])  # hidden layer sizes

        self.online_network = self.build_simple_q_network(
            self.state_size, self.action_size, self.nn_layers
        )
        self.target_network = self.build_simple_q_network(
            self.state_size, self.action_size, self.nn_layers
        )
        self.target_network.set_weights(
            self.online_network.get_weights()
        )  # Start with same weights

        # Optimizer and Loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Counters
        self.global_step_count = (
            0  # Total environment steps across all episodes for this agent
        )
        self.learn_step_counter = 0  # Number of times learn() has been called
        self.tensorboard_writer = tf.summary.create_file_writer(f"logs/{self.agent_id}")

    def build_simple_q_network(self, input_shape, output_shape, hidden_layers):
        """
        Simple feed-forward Q-Network architecture.

        Input: Flattened state vector.
        Output: Q-values for each action.
        """
        model = keras.Sequential(
            [
                Input(shape=(input_shape,)),
            ]
        )
        for layer_size in hidden_layers:
            model.add(
                Dense(
                    layer_size,
                    activation="relu",
                    kernel_initializer=initializers.HeNormal(),
                    bias_initializer=initializers.Zeros(),
                )
            )

        model.add(
            Dense(
                output_shape,
                activation="linear",
                kernel_initializer=initializers.GlorotUniform(),
                bias_initializer=initializers.Zeros(),
            )
        )  # Linear activation - raw probs
        # No compile needed here, compilation happens when defining the training step
        return model

    # * IMPORTANT: experimental, do not use
    def build_branching_q_network(self, state_size, action_size, hidden_layers):
        """
        Q-Network with separate branches for different parts of the state input.
        Assumes state vector structure: [local_queues(12), local_signal(2), presence(4), neighbor_N(14), E(14), S(14), W(14)]
        """
        # Define input layers for different parts of the state
        local_queues_input = Input(shape=(12,), name="local_queues_input")
        local_signal_input = Input(shape=(2,), name="local_signal_input")
        presence_input = Input(shape=(4,), name="presence_input")
        neighbor_n_input = Input(shape=(14,), name="neighbor_n_input")
        neighbor_e_input = Input(shape=(14,), name="neighbor_e_input")
        neighbor_s_input = Input(shape=(14,), name="neighbor_s_input")
        neighbor_w_input = Input(shape=(14,), name="neighbor_w_input")

        # Process each input branch
        local_queues_branch = Dense(16, activation="relu")(local_queues_input)
        local_signal_branch = Dense(8, activation="relu")(local_signal_input)
        presence_branch = Dense(8, activation="relu")(presence_input)
        neighbor_n_branch = Dense(16, activation="relu")(neighbor_n_input)
        neighbor_e_branch = Dense(16, activation="relu")(neighbor_e_input)
        neighbor_s_branch = Dense(16, activation="relu")(neighbor_s_input)
        neighbor_w_branch = Dense(16, activation="relu")(neighbor_w_input)

        # Concatenate the outputs of the branches
        concatenated = Concatenate()(
            [
                local_queues_branch,
                local_signal_branch,
                presence_branch,
                neighbor_n_branch,
                neighbor_e_branch,
                neighbor_s_branch,
                neighbor_w_branch,
            ]
        )

        x = concatenated
        for layer_size in hidden_layers:
            x = Dense(layer_size, activation="relu")(x)

        output_layer = Dense(action_size, activation="linear", name="q_values_output")(
            x
        )

        model = tf.keras.Model(
            inputs=[
                local_queues_input,
                local_signal_input,
                presence_input,
                neighbor_n_input,
                neighbor_e_input,
                neighbor_s_input,
                neighbor_w_input,
            ],
            outputs=output_layer,
        )
        # No compile needed here
        return model

    def select_action(self, state_tensor):
        """
        Selects an action using the epsilon-greedy policy.

        Args:
            state_tensor (tf.Tensor): The current state observation as a TensorFlow tensor
                                     with a batch dimension (shape [1, state_size]).

        Returns:
            int: The chosen action index (0-3).
        """
        # Decay epsilon
        if self.global_step_count < EXPLORATION_ONLY_STEPS:
            self.epsilon = 1.0
        elif self.epsilon > self.epsilon_min:
            self.epsilon = max(
                0.01,
                1.0
                * np.exp(
                    -(self.global_step_count - EXPLORATION_ONLY_STEPS)
                    / EXPLORATION_ONLY_STEPS
                ),
            )

        if np.random.rand() < self.epsilon:
            # Explore: Choose a random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploit: Choose the action with the highest predicted Q-value
            q_values_tensor = self.online_network(state_tensor)  # Shape [1, 4]
            # If using the branching network, need to split input tensor first
            # q_values_tensor = self.online_network(self._split_state_tensor(state_tensor))

            # Get the action with the highest Q-value
            action_index = tf.argmax(q_values_tensor, axis=1).numpy()[0]
            return action_index

    def _split_state_tensor(self, state_tensor):
        """Helper to split the concatenated state tensor for the branching network."""
        # Assumes state_tensor shape is [batch_size, 89]
        # Need to slice the tensor according to the state vector structure
        # [local_queues(12), local_signal(5), presence(4), neighbor_N(17), E(17), S(17), W(17)]
        start = 0
        local_queues = state_tensor[:, start : start + 12]
        start += 12
        local_signal = state_tensor[:, start : start + 2]
        start += 2
        presence = state_tensor[:, start : start + 4]
        start += 4
        neighbor_n = state_tensor[:, start : start + 14]
        start += 14
        neighbor_e = state_tensor[:, start : start + 14]
        start += 14
        neighbor_s = state_tensor[:, start : start + 14]
        start += 14
        neighbor_w = state_tensor[:, start : start + 14]

        return [
            local_queues,
            local_signal,
            presence,
            neighbor_n,
            neighbor_e,
            neighbor_s,
            neighbor_w,
        ]

    def store_experience(self, experience: np.array):
        """Stores an experience tuple (s, a, r, s', d) in the replay buffer."""
        # Experience is expected as (NumPy array, int, float, NumPy array, bool)
        self.replay_buffer.add(experience)
        self.global_step_count += 1  # Increment global step counter here

    def remember(
        self,
        state: np.array,
        action: np.array,
        reward: np.array,
        next_state: np.array,
        done: np.array,
    ):
        """Stores an experience tuple (state, action, reward, next_state, done) in the replay buffer."""
        experience = (state, action, reward, next_state, done)
        logger.debug(
            f"At agent training: state shape {state.shape}, next_state shape {next_state.shape}"
        )
        self.replay_buffer.add(experience)
        # logger.warning(f'Replay buffer size: {self.replay_buffer.__len__()}, {self.agent_id}')

    def learn(self):
        """
        Performs a learning update step using a batch of experiences from the replay buffer.
        """
        # Only learn if enough experiences are in the buffer
        if self.replay_buffer.__len__() < self.batch_size:
            # logger.warning(f"Not enough samples in buffer: {self.replay_buffer.__len__()}, decrease batch size: {self.batch_size}, {self.global_step_count}")
            return  # Not enough data yet
        # logger.info(f'Num samples {self.replay_buffer.__len__()}')
        states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Double DQN: use online network for action selection, target for evaluation
        next_actions = tf.argmax(
            self.online_network(next_states_tf), axis=1, output_type=tf.int32
        )
        indices = tf.stack([tf.range(self.batch_size), next_actions], axis=1)
        target_q_values = tf.gather_nd(self.target_network(next_states_tf), indices)
        target_q_values = tf.cast(target_q_values, dtype=tf.float32)

        targets = rewards_tf + self.gamma * (1.0 - dones_tf) * target_q_values

        with tf.GradientTape() as tape:
            q_values_all = tf.cast(self.online_network(states_tf), tf.float32)
            predicted_q = tf.reduce_sum(
                q_values_all
                * tf.one_hot(actions_tf, self.action_size, dtype=tf.float32),
                axis=1,
            )
            loss = tf.keras.losses.MeanAbsoluteError()(targets, predicted_q)

        # logger.info(f"Loss value before backprop {loss}")

        grads = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.online_network.trainable_variables)
        )

        self.learn_step_counter += 1

        q_values_mean = tf.reduce_mean(q_values_all).numpy()
        q_values_std = tf.math.reduce_std(q_values_all).numpy()
        target_q_values_mean = tf.reduce_mean(target_q_values).numpy()
        target_q_values_std = tf.math.reduce_std(target_q_values).numpy()
        action_distribution = tf.reduce_sum(
            tf.one_hot(actions_tf, self.action_size), axis=0
        ).numpy()

        with self.tensorboard_writer.as_default():
            tf.summary.scalar("loss", loss, step=self.learn_step_counter)
            tf.summary.scalar("epsilon", self.epsilon, step=self.learn_step_counter)
            tf.summary.scalar(
                "q_values_mean_online", q_values_mean, step=self.learn_step_counter
            )
            tf.summary.scalar(
                "q_values_std_online", q_values_std, step=self.learn_step_counter
            )
            tf.summary.scalar(
                "q_values_mean_target", target_q_values_mean, step=self.learn_step_counter
            )
            tf.summary.scalar(
                "q_values_std_target", target_q_values_std, step=self.learn_step_counter
            )
            tf.summary.histogram(
                "action_distribution", action_distribution, step=self.learn_step_counter
            )

        # Soft updates
        # self.update_target_network_soft()

        # Hard updates
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.update_target_network()

        # logger.info(f"Loss before sending: {loss}")
        return loss

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_network.set_weights(self.online_network.get_weights())
        logger.info(
            f"Agent {self.agent_id}: Target network updated at learn step {self.learn_step_counter}"
        )

    @tf.function(reduce_retracing=True)
    def update_target_network_soft(self):
        """Performs soft update of target network weights."""
        online_weights = self.online_network.variables
        target_weights = self.target_network.variables

        for i in range(len(target_weights)):
            # In-place update using tf.Variable.assign()
            target_weights[i].assign(
                self.tau * online_weights[i] + (1.0 - self.tau) * target_weights[i]
            )

    def save_model(self, filepath):
        """Saves the weights of the online network."""
        try:
            self.online_network.save_weights(filepath)
            logger.info(f"Agent {self.agent_id}: Model weights saved to {filepath}")
        except Exception as e:
            logger.exception(
                f"Agent {self.agent_id}: Error saving model weights to {filepath}: {e}"
            )

    def load_model(self, filepath):
        """Loads weights into the online network (and target network)."""
        try:
            self.online_network.load_weights(filepath)
            self.target_network.set_weights(self.online_network.get_weights())
            logger.info(f"Agent {self.agent_id}: Model weights loaded from {filepath}")
            return True
        except Exception as e:
            logger.exception(
                f"Agent {self.agent_id}: Error loading model weights from {filepath}: {e}"
            )
            return False

    def get_epsilon(self) -> float:
        """Returns the current epsilon value."""
        return self.epsilon

    def replay(self) -> float:
        """Performs a learning step using experiences from the replay buffer."""
        loss = self.learn()
        if loss is None:
            return 0  # workaround, in case replay buffer is not filled.
        # todo: do the replay buffer check in training script as well, to prevent this
        return loss
