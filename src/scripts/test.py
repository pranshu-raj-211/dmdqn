"""
Evaluates trained Multi-Agent DQN agents or baselines on the SUMO traffic environment.

Loads trained models, runs episodes with deterministic policies (low/no epsilon)
or baseline logic, collects performance metrics, and provides a comparison summary.
"""

import tensorflow as tf
import numpy as np
import yaml
import argparse
import os
import sys
import time
import pandas as pd
import traci
from collections import deque

from src.agents.dqn_agent import DQNAgent
from src.agents.sumo_env import SumoTrafficEnvironment
from train import set_seeds, load_config


def parse_eval_args():
    """Parses command-line arguments for evaluation."""
    # *bypass this for now, just use hardcoded values/config files
    parser = argparse.ArgumentParser(description="Evaluate MARL Traffic Agents")
    
    parser.add_argument("--agent_config", type=str, default="agent_config.yaml",
                        help="Path to agent hyperparameters YAML file")
    parser.add_argument("--env_config", type=str, default="env_config_3x3.yaml",
                        help="Path to environment and intersections YAML file")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory containing saved agent *.weights.h5 files (REQUIRED for DQN eval)")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                        help="Number of episodes to run evaluation")
    parser.add_argument("--eval_seed_start", type=int, default=10000,
                        help="Starting seed for evaluation episode traffic")
    parser.add_argument("--eval_epsilon", type=float, default=0.01,
                        help="Epsilon value for DQN agent during evaluation (greedy)")
    parser.add_argument("--modes", nargs='+', type=str, default=['dqn', 'random'], # Add 'fixed' if implemented
                        help="Modes to evaluate (dqn, random, fixed)")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv",
                        help="Filename to save detailed evaluation results")
    return parser.parse_args()


def run_evaluation_episode(env, config, episode_seed, mode='dqn', agents=None, eval_epsilon=0.01, fixed_cycle=None):
    """Runs a single evaluation episode for a given mode."""
    print(f"--- Running Eval Episode Seed: {episode_seed}, Mode: {mode.upper()} ---")
    set_seeds(episode_seed)
    observations_np = env.reset(sumo_seed=episode_seed)

    done = False
    step = 0
    episode_rewards = {agent_id: 0 for agent_id in env.get_controlled_intersection_ids()}
    max_metrics_len = config.get('max_steps_per_episode', 1000) * len(env.get_controlled_intersection_ids())
    all_step_queues = deque(maxlen=max_metrics_len)
    # Add deques for delay, throughput etc. if calculated

    # Placeholder: Implement fixed-time state logic if running 'fixed' mode
    fixed_time_state = {}
    if mode == 'fixed' and fixed_cycle:
        for agent_id in env.get_controlled_intersection_ids():
             fixed_time_state[agent_id] = {'phase_idx': 0, 'time_in_phase': 0.0}
             # Ensure cycle definition is valid
             if agent_id not in fixed_cycle or not fixed_cycle[agent_id]:
                  print(f"Warning: Missing or empty fixed cycle definition for {agent_id}")
                  # Handle error: skip fixed mode or use default
    elif mode == 'fixed' and not fixed_cycle:
         print("Error: Fixed mode selected but no fixed_cycle definition provided.")
         return None # Cannot run fixed mode

    while not done:
        step += 1
        actions_dict = {}

        for agent_id in env.get_controlled_intersection_ids():
            action = 0 # Default action
            if mode == 'dqn':
                if agents and agent_id in agents:
                    current_state_np = observations_np[agent_id]
                    state_tensor = tf.convert_to_tensor([current_state_np], dtype=tf.float32)
                    # Use greedy action selection
                    if np.random.rand() < eval_epsilon: # Allow minimal exploration during eval?
                         action = np.random.randint(0, env.get_action_size(agent_id))
                    else:
                         action = agents[agent_id].select_greedy_action(state_tensor)
                else:
                    print(f"Warning: DQN mode but agent {agent_id} not found/loaded.")
                    action = np.random.randint(0, env.get_action_size(agent_id)) # Fallback
            elif mode == 'random':
                action = np.random.randint(0, env.get_action_size(agent_id))
            elif mode == 'fixed':
                # <<< Placeholder: Implement fixed-time action selection logic here >>>
                # Use fixed_time_state and fixed_cycle definition passed in.
                # Example sketch:
                state = fixed_time_state[agent_id]
                cycle_def = fixed_cycle[agent_id]
                current_phase_action_idx, current_dur = cycle_def[state['phase_idx']]

                if state['time_in_phase'] >= current_dur:
                     state['phase_idx'] = (state['phase_idx'] + 1) % len(cycle_def)
                     state['time_in_phase'] = 0.0
                action, _ = cycle_def[state['phase_idx']] # Action index IS the phase index here
                state['time_in_phase'] += config['step_duration'] # Increment by RL step duration
                # Note: This assumes action index directly maps to a phase.
                # A helper might be needed to get the actual phase index in SUMO if complex.

            actions_dict[agent_id] = action

        try:
            next_observations_np, rewards, done, info = env.step(actions_dict)
        except traci.exceptions.TraCIException as e:
            print(f"TraCI error during eval env.step(): {e}. Ending episode.")
            done = True; rewards = {aid: 0 for aid in env.get_controlled_intersection_ids()}; next_observations_np = observations_np
        except Exception as e:
            print(f"Unexpected error during eval env.step(): {e}. Ending episode.")
            done = True; rewards = {aid: 0 for aid in env.get_controlled_intersection_ids()}; next_observations_np = observations_np

        # --- Collect Metrics ---
        for agent_id in env.get_controlled_intersection_ids():
            # Reward
            episode_rewards[agent_id] += rewards.get(agent_id, 0)
            # Queue Length (from state)
            current_queues = observations_np[agent_id][:12] # First 12 elements are queues
            all_step_queues.append(np.sum(current_queues)) # Append sum of queues for this agent this step
            # <<< Placeholder: Add collection for other metrics like delay (from info or TraCI) >>>

        observations_np = next_observations_np
        # Check max steps
        if step >= config.get('max_steps_per_episode', 1000): done = True

    # --- Calculate Episode Averages ---
    avg_step_queue = np.mean(all_step_queues) if all_step_queues else 0
    total_reward = sum(episode_rewards.values())
    avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0

    print(f"--- Episode Seed {episode_seed} Mode {mode.upper()} Finished (Steps: {step}) ---")
    print(f" Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Avg Step Queue Sum: {avg_step_queue:.2f}")

    return {
        "mode": mode,
        "seed": episode_seed,
        "total_reward": total_reward,
        "avg_reward_per_agent": avg_reward,
        "avg_step_queue_sum": avg_step_queue,
        "steps": step,
        # <<< Add other aggregated metrics here >>>
    }

# --- Main Evaluation Function ---
def main_eval(args):
    """Main function to orchestrate evaluation."""
    config = load_config(args.agent_config, args.env_config)
    # Add evaluation specific args to config
    config.update(vars(args))
    print("Evaluation Configuration Loaded.")

    # --- Initialize Environment ---
    print("Initializing SUMO Environment for Evaluation...")
    try:
        env = SumoTrafficEnvironment(
            sumo_cfg_path=config['sumo_cfg_path'],
            controlled_intersections=config['controlled_intersections'],
            step_duration=config['step_duration'],
            max_simulation_time=config.get('max_sim_time'), # Use max sim time from config
            neighbor_padding_value=config.get('neighbor_padding_value', -1.0)
        )
    except Exception as e:
         print(f"Error initializing environment: {e}")
         sys.exit(1)

    agent_ids = env.get_controlled_intersection_ids()
    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # --- Load Trained Agents (if evaluating DQN) ---
    agents = None
    if 'dqn' in args.modes:
        if not args.model_dir or not os.path.isdir(args.model_dir):
            print(f"Error: --model_dir='{args.model_dir}' not found or not specified. Cannot evaluate DQN mode.")
            args.modes.remove('dqn') # Remove dqn if models can't be loaded
        else:
            agents = {}
            print(f"Loading trained models from: {args.model_dir}")
            models_loaded_count = 0
            agent_hyperparams = {k: config.get(k) for k in [ # Get relevant agent params from loaded config
                 "learning_rate", "gamma", "epsilon_start", "epsilon_min",
                 "epsilon_decay_steps", "replay_buffer_size", "batch_size",
                 "target_update_frequency", "nn_layers"
             ]}
            for agent_id in agent_ids:
                agents[agent_id] = DQNAgent(state_size, action_size, agent_id, agent_hyperparams)
                model_path = os.path.join(args.model_dir, f"agent_{agent_id}.weights.h5")
                if agents[agent_id].load_model(model_path):
                    models_loaded_count += 1
                else:
                     print(f"Warning: Failed to load model for {agent_id}. Evaluation might be inaccurate.")
            if models_loaded_count < len(agent_ids):
                 print(f"Warning: Only loaded {models_loaded_count}/{len(agent_ids)} agent models.")

    # --- Define Fixed Cycle (Example - Adapt or load from config!) ---
    fixed_cycle_definition = None
    if 'fixed' in args.modes:
        print("Defining fixed cycle for evaluation...")
        # <<< Placeholder: Define your fixed cycle here, e.g., load from env_config >>>
        fixed_cycle_definition = {}
        if 'controlled_intersections' in config:
            for int_cfg in config['controlled_intersections']:
                # Example: Phase 0 for 30s, Phase 2 for 30s (using action indices from YAML example)
                 fixed_cycle_definition[int_cfg['id']] = [(0, 30.0), (2, 30.0)]
        if not fixed_cycle_definition:
             print("Warning: Could not define fixed cycle. Skipping 'fixed' mode.")
             if 'fixed' in args.modes: args.modes.remove('fixed')


    # --- Run Evaluation Episodes ---
    all_results = []
    print(f"\nStarting Evaluation for modes: {args.modes} ({args.num_eval_episodes} episodes each)...")

    for i in range(args.num_eval_episodes):
        eval_seed = args.eval_seed_start + i
        for mode in args.modes:
            episode_result = run_evaluation_episode(
                env, config, eval_seed, mode=mode, agents=agents,
                eval_epsilon=args.eval_epsilon, fixed_cycle=fixed_cycle_definition
            )
            if episode_result: # Only append if episode ran successfully
                all_results.append(episode_result)
            time.sleep(0.5) # Small delay between episodes/modes

    # --- Cleanup ---
    env.close_sumo()
    print("SUMO Environment Closed.")

    # --- Process and Print Results ---
    if not all_results:
         print("No evaluation results collected.")
         return

    results_df = pd.DataFrame(all_results)
    print("\n--- Evaluation Summary ---")
    summary = results_df.groupby('mode').agg(
        mean_reward=('total_reward', 'mean'),
        std_reward=('total_reward', 'std'),
        mean_avg_queue=('avg_step_queue_sum', 'mean'),
        std_avg_queue=('avg_step_queue_sum', 'std'),
        mean_steps=('steps', 'mean'),
        episodes=('seed', 'count')
    )
    print(summary)

    # Save detailed results to CSV
    try:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nDetailed evaluation results saved to: {args.output_csv}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Ensure SUMO_HOME is set ---
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    args = parse_eval_args()
    main_eval(args)