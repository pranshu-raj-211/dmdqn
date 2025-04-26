"""
Training Script for Multi-Agent Deep Q-Networks (MADQN) in Traffic Signal Control

This script trains agents to optimize traffic signal control using SUMO and DQN.

How to Run:
1. Ensure SUMO is installed and the SUMO_HOME environment variable is set.
2. Install required Python dependencies (e.g., TensorFlow, numpy, traci).
3. Configure the paths to SUMO configuration and network files in the script.
4. Run the script: `python train.py`

Configurations:
- SUMO_CFG_PATH: Path to the SUMO configuration file.
- SUMO_NET_PATH: Path to the SUMO network file.
- EPISODES: Number of training episodes.
- MAX_LANES_PER_DIRECTION: Maximum number of lanes per direction for state representation.
- STEP_DURATION: Duration of each simulation step in seconds.

How It Works:
1. The script initializes the SUMO environment and identifies traffic light junctions.
2. Each junction is assigned a DQN agent.
3. During training:
   - Agents select actions based on their current state.
   - Actions are applied to the SUMO simulation.
   - The environment transitions to the next state, and rewards are calculated.
   - Agents update their policies using the observed transitions.
4. The training loop continues for the specified number of episodes.
"""

import os
import random
import numpy as np
import tensorflow as tf
import traci
import yaml
import wandb

from src.agents.dqn_agent import DQNAgent
from src.experimental.order_lanes import (
    get_traffic_light_junction_ids_and_net_sumolib,
    build_junction_lane_mapping,
    order_lanes_in_edge,
    get_own_state,
    build_state_vector,
)

AGENT_CONFIG_PATH = ""
ENV_CONFIG_PATH = ""
SUMO_CFG_PATH = "src/sumo_files/scenarios/grid_3x3.sumocfg"
SUMO_NET_PATH = "src/sumo_files/scenarios/grid_3x3.net.xml"
baseline = None

# Training parameters
EPISODES = 100
MAX_LANES_PER_DIRECTION = 3
STEP_DURATION = 1.0


def set_seeds(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # control hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Seeds set to: {seed_value}")


def load_config(agent_yaml_path, env_yaml_path):
    """Loads agent and environment configurations from YAML files."""
    if not os.path.exists(agent_yaml_path):
        raise FileNotFoundError(f"Agent config file not found: {agent_yaml_path}")
    if not os.path.exists(env_yaml_path):
        raise FileNotFoundError(f"Environment config file not found: {env_yaml_path}")

    with open(agent_yaml_path, "r") as f:
        agent_config = yaml.safe_load(f)
    with open(env_yaml_path, "r") as f:
        env_config = yaml.safe_load(f)

    # Combine configurations (agent config overrides env config if keys clash)
    config = {**env_config, **agent_config}
    return config


def initialize_environment():
    traci.start(["sumo", "-c", SUMO_CFG_PATH])
    tl_junctions = get_traffic_light_junction_ids_and_net_sumolib(SUMO_NET_PATH)
    all_lanes = traci.lane.getIDList()
    non_internal_lanes = [lane for lane in all_lanes if not lane.startswith(":")]
    junction_lane_map = build_junction_lane_mapping(tl_junctions, non_internal_lanes)
    ordered_junction_lane_map = order_lanes_in_edge(junction_lane_map)
    return tl_junctions, ordered_junction_lane_map


# Create agents for each junction
def create_agents(tl_junctions):
    agents = {}
    # Example configuration for agents
    agent_config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay_steps": 100000,
        "replay_buffer_size": 10000,
        "batch_size": 32,
        "target_update_frequency": 1000,
        "nn_layers": [128, 128],
    }

    for junction_id in tl_junctions:
        agents[junction_id] = DQNAgent(
            state_size=74,  # Example state size
            action_size=2,  # Example action size
            agent_id=junction_id,
            config=agent_config
        )
    return agents


# Initialize wandb
run = wandb.init(
    project="dqn_multi_agent_traffic",
    config={
        "episodes": EPISODES,
        "max_lanes_per_direction": MAX_LANES_PER_DIRECTION,
        "step_duration": STEP_DURATION,
        "sumo_cfg_path": SUMO_CFG_PATH,
        "sumo_net_path": SUMO_NET_PATH,
    },
    settings=wandb.Settings(init_timeout=300),
)
print("logged into wandb")


# Training loop
def train_agents():
    tl_junctions, ordered_junction_lane_map = initialize_environment()
    agents = create_agents(tl_junctions)

    for episode in range(EPISODES):
        print(f"Episode {episode + 1}/{EPISODES}")
        traci.load(["-c", SUMO_CFG_PATH])
        global_state = {}
        current_time = traci.simulation.getTime()

        # Initialize global state
        for junction in tl_junctions:
            global_state[junction] = build_state_vector(
                junction_id=junction,
                tl_junctions=tl_junctions,
                structured_junction_lane_map=ordered_junction_lane_map,
                max_lanes_per_direction=MAX_LANES_PER_DIRECTION,
                current_sim_time=current_time,
                global_state=global_state,
            )

        done = False
        total_reward = 0  # Track total reward for the episode
        while not done:
            actions = {}
            for junction_id, agent in agents.items():
                state = global_state[junction_id]
                # Convert state to TensorFlow tensor before selecting action
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                actions[junction_id] = agent.select_action(state_tensor)

            # Apply actions and step simulation
            for junction_id, action in actions.items():
                traci.trafficlight.setPhase(junction_id, action)

            target_time = current_time + STEP_DURATION
            while current_time < target_time:
                traci.simulationStep()
                current_time = traci.simulation.getTime()

            # Update global state and calculate rewards
            next_global_state = {}
            rewards = {}
            for junction in tl_junctions:
                next_global_state[junction] = get_own_state(
                    junction_id=junction,
                    structured_junction_lane_map=ordered_junction_lane_map,
                    max_lanes_per_direction=MAX_LANES_PER_DIRECTION,
                    current_sim_time=current_time,
                )
                rewards[junction] = calculate_reward(
                    global_state[junction], next_global_state[junction]
                )
                total_reward += rewards[junction]  # Accumulate reward

            # Train agents
            for junction_id, agent in agents.items():
                agent.remember(
                    global_state[junction_id],
                    actions[junction_id],
                    rewards[junction_id],
                    next_global_state[junction_id],
                    done,
                )
                agent.replay()

            global_state = next_global_state

        # Log episode metrics to wandb
        run.log({"episode": episode + 1, "total_reward": total_reward})
        print(f"Episode {episode + 1} complete. Total Reward: {total_reward}")

    traci.close()
    run.finish()


def calculate_reward(current_state, next_state):
    return sum(current_state[:12]) - sum(next_state[:12])


if __name__ == "__main__":
    train_agents()
