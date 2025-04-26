"""
Main script to train Multi-Agent Deep Q-Networks (MADQN) for traffic signal control using SUMO.

Loads configuration, initializes the environment and agents, runs the training loop,
logs metrics to Weights & Biases (WandB), and saves trained models.
Supports running random baseline for comparison.
"""

import tensorflow as tf
import numpy as np
import yaml
import wandb
import argparse
import os
import sys
import time
import traci

from src.agents.dqn_agent import DQNAgent
from src.agents.sumo_env import SumoTrafficEnvironment

import random

def set_seeds(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # control hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    print(f"Seeds set to: {seed_value}")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Multi-Agent DQN for Traffic Control")

    # Configuration Files
    parser.add_argument("--agent_config", type=str, default="agent_config.yaml",
                        help="Path to agent hyperparameters YAML file")
    parser.add_argument("--env_config", type=str, default="env_config_3x3.yaml",
                        help="Path to environment and intersections YAML file")

    # Training Control
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_freq", type=int, default=50, help="Frequency (in episodes) to save models")
    parser.add_argument("--run_name", type=str, default=f"dqn_train_{int(time.time())}",
                        help="Name for this training run")

    # WandB Logging
    parser.add_argument("--wandb_project", type=str, default="marl_traffic_sumo_final",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, required=True,
                        help="Your WandB username or team entity (REQUIRED)") # <<< REQUIRED >>>
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online",
                        help="WandB mode (e.g., 'disabled' for no logging)")

    # Baseline Mode
    parser.add_argument("--baseline", type=str, choices=["random"], default=None,
                        help="Run a baseline instead of training DQN (e.g., 'random')")

    return parser.parse_args()


def load_config(agent_yaml_path, env_yaml_path):
    """Loads agent and environment configurations from YAML files."""
    if not os.path.exists(agent_yaml_path):
        raise FileNotFoundError(f"Agent config file not found: {agent_yaml_path}")
    if not os.path.exists(env_yaml_path):
        raise FileNotFoundError(f"Environment config file not found: {env_yaml_path}")

    with open(agent_yaml_path, 'r') as f:
        agent_config = yaml.safe_load(f)
    with open(env_yaml_path, 'r') as f:
        env_config = yaml.safe_load(f)

    # Combine configurations (agent config overrides env config if keys clash)
    config = {**env_config, **agent_config}
    return config


def main(args):
    """Main function to orchestrate training."""
    set_seeds(args.seed)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} Physical GPU(s). Memory growth enabled.")
      except RuntimeError as e: 
          print(f"Error setting memory growth: {e}")
    else: 
        print("No GPU detected by TensorFlow, using CPU.")

    config = load_config(args.agent_config, args.env_config)
    # Merge argparse arguments into config (command-line overrides YAML)
    config.update(vars(args))
    print("Configuration loaded:")
    # print(yaml.dump(config, indent=2)) # Optional: print full config

    wandb.init(
        project=config['wandb_project'],
        entity=config['wandb_entity'],
        config=config, # Log all hyperparameters
        mode=args.wandb_mode
    )
    cfg = wandb.config

    print("Initializing SUMO Environment...")
    try:
        env = SumoTrafficEnvironment(
            sumo_cfg_path=cfg.sumo_cfg_path,
            controlled_intersections=cfg.controlled_intersections,
            step_duration=cfg.step_duration,
            max_simulation_time=cfg.max_sim_time,
            neighbor_padding_value=cfg.neighbor_padding_value
        )
    except Exception as e:
         print(f"Error initializing environment: {e}")
         wandb.finish()
         sys.exit(1)

    agent_ids = env.get_controlled_intersection_ids()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    print(f"Environment initialized. Agents: {agent_ids}, State size: {state_size}, Action size: {action_size}")

    agents = {}
    if args.baseline is None:
        print("Initializing DQN Agents...")
        agent_hyperparams = {k: getattr(cfg, k) for k in [
            "learning_rate", "gamma", "epsilon_start", "epsilon_min",
            "epsilon_decay_steps", "replay_buffer_size", "batch_size",
            "target_update_frequency", "nn_layers"
        ]}
        for agent_id in agent_ids:
            agents[agent_id] = DQNAgent(state_size, action_size, agent_id, agent_hyperparams)
        print(f"Initialized {len(agents)} DQN agents.")
    else:
        print(f"Running Baseline: {args.baseline.upper()}")

    global_step_counter = 0
    print(f"\nStarting Training for {cfg.num_episodes} episodes...")

    for episode in range(cfg.num_episodes):
        episode_start_time = time.time()
        episode_sumo_seed = cfg.seed + episode
        observations_np = env.reset(sumo_seed=episode_sumo_seed)

        episode_rewards = {agent_id: 0 for agent_id in agent_ids}
        episode_losses = {agent_id: [] for agent_id in agent_ids}
        done = False
        step = 0

        while not done:
            step += 1
            global_step_counter += 1

            actions_dict = {}
            for agent_id in agent_ids:
                current_state_np = observations_np[agent_id]
                state_tensor = tf.convert_to_tensor([current_state_np], dtype=tf.float32)

                if args.baseline == "random":
                    action = np.random.randint(0, action_size)
                elif args.baseline is None: # DQN Agent action
                    action = agents[agent_id].select_action(state_tensor)
                else:
                    # Placeholder for other baselines if added
                    action = 0
                actions_dict[agent_id] = action

            try:
                next_observations_np, rewards, done, info = env.step(actions_dict)
            except traci.exceptions.TraCIException as e:
                print(f"TraCI error during env.step(): {e}. Ending episode.")
                done = True # End episode if SUMO connection fails
                rewards = {agent_id: 0 for agent_id in agent_ids} # Assign zero reward
                next_observations_np = observations_np # Use last valid observation
            except Exception as e:
                 print(f"Unexpected error during env.step(): {e}. Ending episode.")
                 done = True
                 rewards = {agent_id: 0 for agent_id in agent_ids}
                 next_observations_np = observations_np


            current_sim_time = env.current_time
            step_log = {
                "global_step": global_step_counter,
                "simulation_time": current_sim_time,
                "episode": episode,
                "step_in_episode": step,
            }
            if args.baseline is None: # DQN Training Mode
                step_losses = {}
                for agent_id in agent_ids:
                    s = observations_np[agent_id]
                    a = actions_dict[agent_id]
                    r = rewards[agent_id]
                    s_prime = next_observations_np[agent_id]
                    d = done

                    experience = (s, a, r, s_prime, d)
                    agents[agent_id].store_experience(experience)
                    loss = agents[agent_id].learn()

                    episode_rewards[agent_id] += r
                    step_log[f"reward_{agent_id}"] = r
                    if loss is not None:
                        loss_val = loss.numpy()
                        episode_losses[agent_id].append(loss_val)
                        step_losses[f"loss_{agent_id}"] = loss_val

                step_log.update(step_losses)
                wandb.log(step_log)

            else:
                 for agent_id in agent_ids:
                     episode_rewards[agent_id] += rewards.get(agent_id, 0)
                     step_log[f"reward_{agent_id}"] = rewards.get(agent_id, 0)
                 wandb.log(step_log)
            observations_np = next_observations_np

            if step >= cfg.max_steps_per_episode:
                print(f"Episode {episode} reached max steps ({cfg.max_steps_per_episode}).")
                done = True # Force end episode

        # Episode End Logging
        episode_duration = time.time() - episode_start_time
        episode_log = {"episode": episode, "episode_duration_sec": episode_duration}
        total_episode_reward = 0
        print(f"--- Episode {episode} Summary (Duration: {episode_duration:.2f}s) ---")

        for agent_id in agent_ids:
            ep_rew = episode_rewards[agent_id]
            total_episode_reward += ep_rew
            episode_log[f"total_reward_{agent_id}"] = ep_rew
            print(f" Agent {agent_id}: Reward={ep_rew:.2f}", end="")
            if args.baseline is None:
                 avg_loss = np.mean(episode_losses[agent_id]) if episode_losses[agent_id] else 0
                 epsilon = agents[agent_id].get_epsilon()
                 episode_log[f"avg_loss_{agent_id}"] = avg_loss
                 episode_log[f"epsilon_{agent_id}"] = epsilon
                 print(f", Avg Loss={avg_loss:.4f}, Epsilon={epsilon:.3f}")
            else:
                print("")

        episode_log["total_reward_all_agents"] = total_episode_reward
        episode_log["steps_in_episode"] = step
        episode_log["avg_reward_all_agents"] = total_episode_reward / len(agent_ids) if agent_ids else 0
        wandb.log(episode_log)
        print(f" Sum Reward: {total_episode_reward:.2f}, Avg Reward: {episode_log['avg_reward_all_agents']:.2f}, Steps: {step}, Sim Time: {current_sim_time:.2f}\n")

        if args.baseline is None and (episode + 1) % cfg.save_freq == 0:
            save_dir = os.path.join("runs", wandb.run.id, f"models_ep_{episode + 1}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving models at episode {episode + 1} to {save_dir}")
            for agent_id, agent in agents.items():
                agent.save_model(os.path.join(save_dir, f"agent_{agent_id}.weights.h5"))

    print("Training finished.")
    env.close_sumo()
    wandb.finish()
    print("SUMO closed, WandB run finished.")


if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    args = parse_args()
    main(args)