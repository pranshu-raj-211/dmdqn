import os
import random
import numpy as np
import traci
import yaml
import wandb

from src.experimental.order_lanes import (
    get_traffic_light_junction_ids_and_net_sumolib,
    build_junction_lane_mapping,
    order_lanes_in_edge,
    get_own_state,
)
from log_config import logger

AGENT_CONFIG_PATH = ""
ENV_CONFIG_PATH = ""
SUMO_CFG_PATH = "src/sumo_files/scenarios/grid_3x3_lefthand/grid_3x3_lht.sumocfg"
SUMO_NET_PATH = "src/sumo_files/scenarios/grid_3x3_lefthand/grid_3x3_lht.net.xml"
baseline = None

EPISODES = 1
MAX_LANES_PER_DIRECTION = 3
STEP_DURATION = 20.0
ACTION_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
MAX_SIM_TIME = 3600


def set_seeds(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    # tf.random.set_seed(seed_value)
    # control hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    logger.info(f"Seeds set to: {seed_value}")


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


run = wandb.init(
    project="dqn_multi_agent_traffic",
    config={
        "baseline": True,
        "baseline_type": "random",
        "episodes": EPISODES,
        "max_lanes_per_direction": MAX_LANES_PER_DIRECTION,
        "step_duration": STEP_DURATION,
        "sumo_cfg_path": SUMO_CFG_PATH,
        "sumo_net_path": SUMO_NET_PATH,
    },
    settings=wandb.Settings(init_timeout=90, mode="online"),
)


class SmoothedValue:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value

    def get_value(self):
        return self.value


def calculate_local_reward(current_state, next_state):
    logger.info(f'reward calc: {current_state[:12]}')
    return -1.0 * sum(current_state[:12])


def calculate_global_reward(global_state: dict, next_global_state: dict):
    """Return negative of difference of total queue length between next state and current."""
    return -1.0 * sum(sum(state[:12]) for state in global_state.values())


def calculate_rewards(
    junction_id: str,
    global_state: dict,
    next_global_state: dict,
    alpha: float,
    beta: float,
) -> float:
    local_reward = calculate_local_reward(
        global_state[junction_id], next_global_state[junction_id]
    )
    global_reward = calculate_global_reward(global_state, next_global_state)
    return alpha * local_reward + beta * global_reward


def random_baseline():
    """Randomly selects actions for traffic lights and runs the simulation."""
    set_seeds(42)
    tl_junctions, ordered_junction_lane_map = initialize_environment()

    for episode in range(EPISODES):
        global_state = dict()
        current_time = traci.simulation.getTime()

        for junction in tl_junctions:
            global_state[junction] = get_own_state(
                junction_id=junction,
                structured_junction_lane_map=ordered_junction_lane_map,
                max_lanes_per_direction=MAX_LANES_PER_DIRECTION,
                current_sim_time=current_time,
            )
            logger.info(f'state for {junction}: {global_state[junction]}')

        done = False
        step_count = 0
        total_reward = 0
        # logger.info(f'step: {step_count}, global state.shape: {len(global_state.values())}, {len(global_state["J_0_0"])}')

        while not done:
            for tl in tl_junctions:
                random_phase = random.choice(list(ACTION_MAP.values()))
                traci.trafficlight.setPhase(tl, random_phase)
            traci.simulationStep()

            # step till the time we're supposed to take next action
            target_time = current_time + STEP_DURATION
            # logger.info(f'current_time: {current_time}, target_time: {target_time}')
            while current_time < target_time:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                done = (
                    traci.simulation.getMinExpectedNumber() == 0
                    or current_time >= MAX_SIM_TIME
                )

            next_global_state = dict()
            rewards = dict()
            global_reward = calculate_global_reward(global_state, next_global_state)
            smooth_global_reward.update(global_reward)
            
            for junction in tl_junctions:
                global_state[junction] = get_own_state(
                    junction_id=junction,
                    structured_junction_lane_map=ordered_junction_lane_map,
                    max_lanes_per_direction=MAX_LANES_PER_DIRECTION,
                    current_sim_time=current_time,
                )
                logger.info(f'state for {junction}: {global_state[junction]}')

                local_reward = calculate_local_reward(
                    global_state[junction], []
                )
                rewards[junction] = 0.3 * local_reward + 0.7 * global_reward
            total_reward = sum(rewards.values())
            logger.warning(f'step: {step_count}, global reward: {global_reward}, total_reward: {total_reward}')
            smooth_total_reward.update(total_reward)
            smooth_global_reward.update(global_reward)
            run.log(
                {
                    "step": step_count,
                    "global_reward": global_reward,
                    "total_reward": total_reward,
                    "smoothed_global_reward": smooth_global_reward.get_value(),
                    "smoothed_total_reward": smooth_total_reward.get_value(),
                }
            )
            step_count += 1

    traci.close()
    logger.info("Random baseline simulation completed.")


def get_queue_length_junction(ordered_junction_lane_map: dict):
    queue_lengths_map = dict()
    for junction, direction_list in ordered_junction_lane_map.items():
        queue = 0
        queue_lengths_map[junction] = list()
        for lane in direction_list:
            queue += traci.lane.getLastStepHaltingNumber(lane)
        queue_lengths_map[junction].append(queue)
    return queue_lengths_map


def timed_baseline():
    """Cycles through traffic light phases with fixed durations."""
    set_seeds(42)  # Ensure reproducibility
    tl_junctions, _ = initialize_environment()

    phase_durations = [10, 10, 10, 10]  # Example fixed durations for 4 phases

    for step in range(int(MAX_SIM_TIME / STEP_DURATION)):
        for tl in tl_junctions:
            current_phase = traci.trafficlight.getPhase(tl)
            next_phase = (current_phase + 1) % len(phase_durations)
            traci.trafficlight.setPhase(tl, next_phase)
        traci.simulationStep()

    traci.close()
    logger.info("Timed baseline simulation completed.")


smooth_global_reward = SmoothedValue(alpha=0.3)
smooth_total_reward = SmoothedValue(alpha=0.3)


if __name__ == "__main__":
    random_baseline()
    # timed_baseline()
