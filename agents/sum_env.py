import os
import sys
import subprocess
import traci
import time
import numpy as np


# Ensure SUMO_HOME is set and TraCI is in the Python path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

class SumoTrafficEnvironment:
    """
    Traffic environment using SUMO and TraCI, designed for collaborative RL.
    Handles simulation, state observation, action execution, and reward.
    Outputs state as NumPy arrays, ready for conversion to TensorFlow tensors.
    """
    def __init__(self, sumo_cfg_path, controlled_intersections, step_duration=1.0, max_simulation_time=3600):
        """
        Initializes the traffic environment.

        Args:
            sumo_cfg_path (str): Path to the SUMO configuration file (.sumocfg).
            controlled_intersections (list): List of dictionaries, each defining a controlled intersection.
                                             Example: [{"id": "intersection_1", "lanes": [...], "tl_id": "...", "action_phases": [...]}]
            step_duration (float): Duration of one RL step in seconds.
            max_simulation_time (float): Maximum simulation time before episode ends.
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.step_duration = step_duration
        self.max_simulation_time = max_simulation_time

        self.sumo_process = None # To hold the SUMO subprocess
        self.current_time = 0.0

        # Store configuration for controlled intersections
        self.controlled_intersections_config = {int_config["id"]: int_config for int_config in controlled_intersections}
        self.controlled_intersection_ids = list(self.controlled_intersections_config.keys())

        # --- Derived from controlled_intersections_config ---
        # Map intersection ID to the IDs of incoming lanes to observe queues
        self.observed_lanes = {
            int_id: config.get("lanes", []) for int_id, config in self.controlled_intersections_config.items()
        }
         # Map intersection ID to its traffic light ID in SUMO
        self.traffic_light_ids = {
            int_id: config.get("tl_id", int_id) for int_id, config in self.controlled_intersections_config.items()
        }
        # Map intersection ID and action index to a SUMO traffic light phase string
        self.action_to_sumo_phase = {
            int_id: config.get("action_phases", {}) for int_id, config in self.controlled_intersections_config.items()
        }

        # --- Placeholder for Neighbor Mapping ---
        # This would ideally be built by analyzing the network geometry after loading in SUMO
        # For now, we'll need a way to define this, perhaps in the controlled_intersections config
        self.neighbor_map = self._build_neighbor_map() # Needs implementation

        # Define placeholder value for padded neighbor info
        self.padding_value = -1.0 # Use a value outside the normal range of traffic data

        # Define the structure/size of neighbor information block
        # Based on our state definition: 12 lane queues + 2 signal state = 14 elements per neighbor slot
        self.neighbor_info_size = 12 + 2 # Assuming neighbor state is same structure as local state for simplicity

        # Calculate the total size of the state vector for one agent
        # 12 (local queues) + 2 (local signal) + 4 (neighbor presence) + 4 * neighbor_info_size
        self.state_vector_size = 12 + 2 + 4 + 4 * self.neighbor_info_size # 14 + 4 + 56 = 74

        # Store previous state to calculate reward based on changes (optional, but common)
        self._previous_observations = None


    def start_sumo(self, use_gui=False):
        """Starts the SUMO simulation as a subprocess and connects TraCI."""
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path]

        print(f"Starting SUMO: {' '.join(sumo_cmd)}")
        try:
            # Start SUMO and wait briefly
            # stdout=sys.stdout, stderr=sys.stderr helps see SUMO errors
            self.sumo_process = subprocess.Popen(sumo_cmd, stdout=sys.stdout, stderr=sys.stderr)
            time.sleep(1) # Give SUMO a moment to start
            traci.init()
            print("TraCI connected.")
        except Exception as e:
            print(f"Error starting SUMO or connecting TraCI: {e}")
            if self.sumo_process:
                 self.sumo_process.terminate()
            sys.exit(1)

    def close_sumo(self):
        """Closes the TraCI connection and terminates the SUMO subprocess."""
        if traci.isconnected():
            traci.close()
            print("TraCI connection closed.")
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()
            print("SUMO process terminated.")

    def reset(self):
        """
        Resets the environment for a new episode.
        Starts SUMO (if not already running), loads the scenario, and returns initial observations.

        Returns:
            dict: A dictionary where keys are agent IDs and values are their initial state observations (as NumPy arrays).
        """
        print("\nResetting environment for a new episode...")
        if traci.isconnected():
             self.close_sumo()
             time.sleep(1)

        self.start_sumo(use_gui=False) # Set to True to watch the simulation

        self.current_time = 0.0

        # Advance simulation to load initial vehicles if needed (e.g., first few steps)
        # traci.simulationStep() # Step once to let vehicles appear

        # Get initial observations for all agents
        initial_observations = self._get_observations()
        self._previous_observations = initial_observations # Store initial state as previous

        print("Environment reset complete.")
        return initial_observations

    def step(self, actions):
        """
        Advances the environment by one time step based on the actions taken by all agents.

        Args:
            actions (dict): A dictionary where keys are agent IDs and values are the
                            action index (0-3) taken by that agent.

        Returns:
            tuple: (next_states, rewards, done, info)
                   - next_states (dict): New state observations (NumPy arrays).
                   - rewards (dict): Rewards received by each agent.
                   - done (bool): True if the episode is finished.
                   - info (dict): Additional information (optional).
        """
        # print(f"\nStepping environment at time {self.current_time}...")
        # print(f"Received actions: {actions}")

        # Store current observations before stepping for reward calculation
        current_observations = self._get_observations() # Or use self._previous_observations if updated at end of step

        # --- Apply actions to SUMO traffic lights ---
        self._apply_actions(actions)

        # --- Advance SUMO simulation ---
        # Advance by step_duration seconds
        steps_to_advance = max(1, int(self.step_duration / traci.simulation.getDeltaT()))
        for _ in range(steps_to_advance):
             traci.simulationStep()
             # Check for episode done within substeps if needed
             if self._is_episode_done_sumo(): # Check SUMO's internal termination
                 break # Exit substep loop early if SUMO finished

        self.current_time = traci.simulation.getTime() # Get actual simulation time

        # --- Get next state observations ---
        next_observations = self._get_observations()

        # --- Calculate rewards ---
        # Calculate reward based on the transition from current_observations to next_observations
        rewards = self._calculate_rewards(current_observations, next_observations, actions)

        # --- Check if episode is done ---
        done = self._is_episode_done()

        info = {}

        # Update previous observations for the next step
        self._previous_observations = next_observations

        # print(f"Step complete. New time: {self.current_time:.2f}, Done: {done}")
        return next_observations, rewards, done, info

    def _apply_actions(self, actions):
        """
        Applies the agents' chosen actions to the SUMO traffic lights via TraCI.
        Translates action index to SUMO phase string and sets the phase.
        """
        # print("Applying actions...")
        for intersection_id, action_index in actions.items():
            if intersection_id in self.traffic_light_ids:
                tl_id = self.traffic_light_ids[intersection_id]
                action_phases = self.action_to_sumo_phase.get(intersection_id, {})
                sumo_phase_string = action_phases.get(action_index)

                if sumo_phase_string:
                    try:
                        # Find the index of the desired phase string in the current SUMO TL program
                        # This is important because setPhase() expects the index, not the string
                        current_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0] # Get the current program logic
                        phase_index_in_sumo = -1
                        for i, phase in enumerate(current_logic.phases):
                            if phase.state == sumo_phase_string:
                                phase_index_in_sumo = i
                                break

                        if phase_index_in_sumo != -1:
                             # Set the traffic light phase
                             traci.trafficlight.setPhase(tl_id, phase_index_in_sumo)
                             # print(f"  Set TL {tl_id} to phase index: {phase_index_in_sumo} ({sumo_phase_string})")
                        else:
                             print(f"Warning: SUMO phase string '{sumo_phase_string}' not found in TL {tl_id}'s current logic.")

                    except traci.exceptions.TraCIException as e:
                         print(f"Error setting phase for TL {tl_id}: {e}")
                else:
                     print(f"Warning: No SUMO phase defined for action index {action_index} at {intersection_id}")


    def _get_phase_index_from_string(self, tl_id, phase_string):
        """Helper to find the index of a phase string within the TL's program."""
        # This is a bit hacky; a better way is to know the phase index directly
        # or define actions by index. But if you only have phase strings...
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0] # Get the current program logic
        for i, phase in enumerate(logic.phases):
            if phase.state == phase_string:
                return i
        # Fallback: if not found, maybe just return 0 or raise error
        return 0 # Or handle error


    def _get_observations(self):
        """
        Collects the current state observation for each controlled intersection.
        Includes local and padded neighbor information.

        Returns:
            dict: A dictionary where keys are agent IDs and values are their
                  state observations as NumPy arrays, ready for NN input.
        """
        # print("Getting observations for all agents...")
        observations = {}
        # First, collect all local observations and neighbor presence
        local_and_presence_data = {}
        for agent_id in self.controlled_intersection_ids:
            local_obs = self._get_local_observation(agent_id) # List of 12 queues + 2 signal state
            neighbor_presence = self._get_neighbor_presence_vector(agent_id) # List of 4 (0 or 1)
            local_and_presence_data[agent_id] = {
                "local": local_obs,
                "presence": neighbor_presence
            }

        # Now, build the full state for each agent, including padded neighbor info
        for agent_id in self.controlled_intersection_ids:
            local_data = local_and_presence_data[agent_id]["local"]
            presence_data = local_and_presence_data[agent_id]["presence"]

            neighbor_info_blocks = []
            # Define a consistent order for neighbors (N, E, S, W)
            neighbor_order = ["N", "E", "S", "W"] # Assuming you can map directions to neighbor IDs

            for direction in neighbor_order:
                 neighbor_id = self._get_neighbor_id_in_direction(agent_id, direction) # Needs implementation
                 if neighbor_id and neighbor_id in self.controlled_intersection_ids:
                      # Neighbor exists and is controlled, get its local data
                      neighbor_local_data = local_and_presence_data[neighbor_id]["local"]
                      neighbor_info_blocks.extend(neighbor_local_data) # Add the neighbor's local state
                 else:
                      # Neighbor does not exist or is not controlled, add padding
                      neighbor_info_blocks.extend([self.padding_value] * self.neighbor_info_size)

            # Combine all parts into the final state vector for this agent
            state_vector_list = local_data + presence_data + neighbor_info_blocks
            observations[agent_id] = np.array(state_vector_list, dtype=np.float32) # Convert to NumPy array

        # print("Observations collected and formatted.")
        return observations

    def _get_local_observation(self, agent_id):
        """
        Collects local state information for a specific agent's intersection using TraCI.
        Returns: List of 12 queue counts + 2 signal state values.
        """
        # print(f"  Getting local observation for {agent_id}...")
        local_obs = []
        # 1. Get Queue Counts (12 values)
        # Need to know the specific lane IDs for each direction and type at this intersection
        # Example: Assuming self.observed_lanes[agent_id] lists lanes in a fixed order
        for lane_id in self.observed_lanes.get(agent_id, []):
             try:
                 queue_len = traci.lane.getWaitingVehicles(lane_id) # Get number of waiting vehicles
                 local_obs.append(queue_len)
             except traci.exceptions.TraCIException:
                 local_obs.append(0.0) # Handle potential errors

        # Ensure we have exactly 12 queue values (pad with 0 if less, though config should prevent this)
        while len(local_obs) < 12:
             local_obs.append(0.0)
        local_obs = local_obs[:12] # Trim if somehow more than 12

        # 2. Get Current Signal State (2 values)
        tl_id = self.traffic_light_ids.get(agent_id)
        if tl_id:
             try:
                 current_phase_index = traci.trafficlight.getPhase(tl_id)
                 time_in_phase = traci.trafficlight.getPhaseDuration(tl_id) - traci.trafficlight.getNextSwitch(tl_id)
                 local_obs.extend([current_phase_index, time_in_phase])
             except traci.exceptions.TraCIException:
                 local_obs.extend([0.0, 0.0]) # Handle potential errors
        else:
             local_obs.extend([0.0, 0.0]) # No traffic light

        return local_obs # This list should have size 14 (12+2)

    def _get_neighbor_presence_vector(self, agent_id):
        """
        Generates the 4-element one-hot vector indicating neighbor presence (N, E, S, W).
        Needs self.neighbor_map and mapping directions to neighbor IDs.
        """
        presence_vector = [0] * 4 # [N_present, E_present, S_present, W_present]
        # Need to map directions to the neighbor_map structure
        # Example: Assuming neighbor_map[agent_id] gives a list of neighbor IDs
        # And you have a way to know which direction each neighbor is.
        # This requires careful setup based on your network geometry.

        # Placeholder logic: Assume neighbor_map tells us which directions have neighbors
        neighbors_in_directions = self._get_neighbors_by_direction(agent_id) # Needs implementation

        if neighbors_in_directions.get("N"): presence_vector[0] = 1
        if neighbors_in_directions.get("E"): presence_vector[1] = 1
        if neighbors_in_directions.get("S"): presence_vector[2] = 1
        if neighbors_in_directions.get("W"): presence_vector[3] = 1

        return presence_vector # This list should have size 4

    def _get_neighbors_by_direction(self, agent_id):
        """
        (Placeholder) Returns a dict mapping directions (N, E, S, W) to neighbor IDs.
        Requires analyzing network geometry or having this info in config.
        """
        # This is complex and depends heavily on your network definition.
        # You'd need to find which outgoing lanes from neighbors connect to incoming lanes at agent_id,
        # and determine the direction.
        # For a grid, you can often infer based on ID naming or coordinates.
        # Example: Dummy implementation
        neighbors_by_dir = {}
        # if agent_id == "intersection_1": # Example for a specific intersection
        #      neighbors_by_dir["E"] = "intersection_2"
        #      neighbors_by_dir["S"] = "intersection_3"
        # ...
        return neighbors_by_dir # Example: {"N": "neighbor_N_id", "E": None, "S": "neighbor_S_id", "W": None}


    def _build_neighbor_map(self):
        """
        (Placeholder) Builds a map showing neighbors for each intersection.
        More complex than the minimal version, needs directionality.
        """
        # This needs to be implemented based on your network structure.
        # It should populate self.neighbor_map and potentially be used by _get_neighbors_by_direction
        # Example: self.neighbor_map = {"int_id_1": ["int_id_2", "int_id_3"], ...}
        # And self._get_neighbors_by_direction would use this map + geometry info.
        print("Building neighbor map (placeholder)...")
        return {} # Replace with actual map


    def _calculate_rewards(self, previous_states, current_states, actions):
        """
        Calculates the reward for each agent based on the state transition.
        Using negative sum of waiting vehicles change.

        Args:
            previous_states (dict): State observations *before* the step (NumPy arrays).
            current_states (dict): State observations *after* the step (NumPy arrays).
            actions (dict): Actions taken in the previous step.

        Returns:
            dict: A dictionary where keys are agent IDs and values are their rewards.
        """
        # print("Calculating rewards...")
        rewards = {}
        for agent_id in self.controlled_intersection_ids:
            # Get the queue lengths from the state vectors (first 12 elements)
            # Note: These are the raw queue counts, not influenced by padding
            # Use .get() with a default to handle cases where an agent might be missing (shouldn't happen with correct logic)
            prev_state_np = previous_states.get(agent_id, np.zeros(self.state_vector_size, dtype=np.float32))
            curr_state_np = current_states.get(agent_id, np.zeros(self.state_vector_size, dtype=np.float32))

            prev_queues = prev_state_np[:12]
            curr_queues = curr_state_np[:12]

            # Reward is the reduction in total queue length at this intersection
            reward = np.sum(prev_queues) - np.sum(curr_queues)

            # --- Optional: Add penalty for neighbor queues ---
            # This requires getting neighbor queue info *after* the step
            # and comparing it to some baseline or previous state.
            # This makes the reward function collaborative.
            # Example (conceptual):
            # neighbor_penalty = 0
            # neighbors = self._get_neighbors_by_direction(agent_id) # Needs implementation
            # for direction, neighbor_id in neighbors.items():
            #      if neighbor_id and neighbor_id in self.controlled_intersection_ids:
            #           # Get queues on lanes *from* neighbor *to* agent_id in current_states
            #           # This requires careful indexing into the neighbor's state block
            #           # or querying TraCI for specific connecting lanes.
            #           # For simplicity, let's skip this part in the minimal implementation.
            #           pass # Add logic here

            # rewards[agent_id] = reward - neighbor_penalty # Include penalty if implemented

            rewards[agent_id] = reward # Use simple local reward for now

        # print("Rewards calculated.")
        return rewards


    def _is_episode_done(self):
        """
        Checks if the current simulation episode should end.
        Combines max time and potentially other conditions.
        """
        return self.current_time >= self.max_simulation_time or self._is_episode_done_sumo()

    def _is_episode_done_sumo(self):
        """Checks if SUMO's internal simulation has finished."""
        try:
            # getMinExpectedNumber returns the total number of vehicles still in the network
            # that are expected to arrive at their destination.
            # If this is 0 and there are no vehicles currently in the network, simulation is done.
            return traci.simulation.getMinExpectedNumber() <= 0 and len(traci.vehicle.getIDList()) == 0
        except traci.exceptions.TraCIException:
            # Handle case where connection might have closed unexpectedly
            return True # Assume done if TraCI call fails


    def get_controlled_intersection_ids(self):
        """Returns a list of IDs for controlled intersections (agents)."""
        return list(self.controlled_intersections_config.keys())

    def get_state_size(self):
        """Returns the size of the state vector for one agent."""
        return self.state_vector_size

    def get_action_size(self, intersection_id):
        """Returns the size of the action space for a specific agent."""
        # Based on our definition of 4 actions per intersection
        return 4


# --- How to use this class ---
if __name__ == "__main__":
    # Import TensorFlow here if you need to demonstrate tensor conversion
    import tensorflow as tf

    # --- Configuration ---
    # **IMPORTANT**: Update this path to your actual SUMO config file
    your_sumo_cfg_file = "path/to/your/simple_grid.sumocfg" # <-- **UPDATE THIS PATH**

    # **IMPORTANT**: Define your controlled intersections based on your SUMO network
    # You need the intersection ID, the IDs of the 12 incoming lanes to observe queues on,
    # the traffic light ID, and the mapping from action index (0-3) to SUMO phase string.
    # You'll find lane IDs and phase strings in your .net.xml file.
    controlled_intersections_config = [
        {
            "id": "intersection_1", # SUMO junction ID
            "lanes": ["lane_id_1", "lane_id_2", ..., "lane_id_12"], # 12 incoming lane IDs
            "tl_id": "intersection_1", # SUMO traffic light ID (often same as junction ID)
            "action_phases": { # Map action index to SUMO phase string
                0: "GGgrrrGGgrrr", # Example: NS Straight Green
                1: "rrGGgrrrGGgr", # Example: EW Straight Green
                2: "rrrGGGrrrrrr", # Example: NS Left Green
                3: "rrrrrrGGGrrr", # Example: EW Left Green
            }
        },
        # Add configurations for other controlled intersections in your network
        # { "id": "intersection_2", "lanes": [...], "tl_id": "...", "action_phases": {...} },
        # ...
    ]


    # 1. Initialize the environment
    env = SumoTrafficEnvironment(
        sumo_cfg_path=your_sumo_cfg_file,
        controlled_intersections=controlled_intersections_config,
        step_duration=5.0, # RL step = 5 seconds of simulation time
        max_simulation_time=1000 # Example: Run for 1000 seconds of simulation time
    )

    # 2. Reset the environment to start an episode
    # This will start SUMO and connect TraCI
    initial_states_np = env.reset() # These are NumPy arrays

    print("\nInitial States (NumPy arrays):")
    for agent_id, state_np in initial_states_np.items():
        print(f"  {agent_id}: shape {state_np.shape}, first 5 elements: {state_np[:5]}")

        # --- Example: Convert NumPy state to TensorFlow Tensor ---
        state_tensor = tf.convert_to_tensor(state_np, dtype=tf.float32)
        # Add a batch dimension (models expect input in batches)
        state_tensor = tf.expand_dims(state_tensor, 0) # Shape becomes [1, state_size]
        print(f"  {agent_id} state as TensorFlow Tensor: shape {state_tensor.shape}, dtype {state_tensor.dtype}")
        # You would feed state_tensor into your Keras model


    # 3. Run the simulation loop for one episode
    done = False
    step_count = 0
    current_states_np = initial_states_np # Keep track of current states (NumPy arrays)

    while not done:
        step_count += 1
        # --- Agent Action Selection (Placeholder) ---
        # In a real RL setup, your agent(s) (Keras models) would use 'current_states_np'
        # to choose actions using their learned policy (epsilon-greedy based on Q-values).
        # You would convert current_states_np to tensors here before feeding to the model.

        actions = {}
        for int_id in env.get_controlled_intersection_ids():
             # Convert agent's current state (NumPy) to TensorFlow tensor
             state_tensor_agent = tf.convert_to_tensor(current_states_np[int_id], dtype=tf.float32)
             state_tensor_agent = tf.expand_dims(state_tensor_agent, 0) # Add batch dim

             # --- Placeholder: Get Q-values from agent's Keras model ---
             # Assuming you have a Keras model 'agent_q_model' for this agent
             # q_values_tensor = agent_q_model(state_tensor_agent) # Shape [1, 4]
             # q_values_numpy = q_values_tensor.numpy()[0] # Convert back to NumPy for action selection

             # --- Action Selection (e.g., Epsilon-Greedy) ---
             # Based on q_values_numpy or a random choice
             action_size = env.get_action_size(int_id)
             # For this example, still using random actions:
             chosen_action_index = np.random.randint(0, action_size) # <-- Replace with agent's action later

             actions[int_id] = chosen_action_index # Action is just the index

        # print(f"\n--- Step {step_count} at time {env.current_time:.2f} ---")
        # print(f"Chosen actions: {actions}")

        # Step the environment forward
        next_states_np, rewards, done, info = env.step(actions) # env returns NumPy arrays

        # --- Agent Learning (Placeholder) ---
        # Your RL agent(s) would use (current_states_np, actions, rewards, next_states_np, done)
        # to perform a learning update here using Keras/TensorFlow.
        # Convert data to tensors for the learning step.
        # Example:
        # for agent_id in env.get_controlled_intersection_ids():
        #      s = tf.convert_to_tensor(current_states_np[agent_id], dtype=tf.float32)
        #      a = actions[agent_id] # Action index
        #      r = rewards[agent_id]
        #      s_prime = tf.convert_to_tensor(next_states_np[agent_id], dtype=tf.float32)
        #      d = done # Or done for this specific agent if using individual done flags

        #      # Perform DQN update using s, a, r, s', d and Keras models (online and target)
        #      # ... calculate Target Q using target model ...
        #      # ... calculate loss using online model ...
        #      # ... apply gradients ...


        # Update current state for the next loop iteration
        current_states_np = next_states_np

        # Print step results (simplified)
        # print("Next States (partial):", {k: v.shape for k, v in next_states_np.items()}) # Print shapes instead of full arrays
        print(f"Step {step_count}: Time {env.current_time:.2f}, Rewards: {rewards}, Done: {done}")

        # Optional: Add a small delay if watching the GUI (set use_gui=True in reset)
        # time.sleep(0.05) # Slightly shorter delay


    print("\nEpisode finished.")

    # 4. Close the SUMO connection and process
    env.close_sumo()