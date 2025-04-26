import os
import sys
import subprocess
import traci
import time
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    if not os.path.exists(tools):
        sys.exit(
            f"Error: SUMO tools directory not found at '{tools}'. Please verify SUMO_HOME."
        )
else:
    print(
        "Warning: SUMO_HOME environment variable not found. Trying default '/usr/share/sumo'."
    )
    default_sumo_home = "/usr/share/sumo"
    if os.path.exists(default_sumo_home):
        os.environ["SUMO_HOME"] = default_sumo_home
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
        if not os.path.exists(tools):
            sys.exit(
                f"Error: SUMO tools directory not found at '{tools}' even after attempting default."
            )
    else:
        sys.exit(
            "Error: Could not find SUMO_HOME variable or a default SUMO installation."
        )

try:
    import sumolib
except ImportError:
    sys.exit("Error: Failed to import 'sumolib'. Make sure SUMO_HOME is set correctly.")

DEFAULT_MAX_LANES_PER_DIRECTION = 3


class SumoTrafficEnvironment:
    """
    SUMO Traffic Environment for Reinforcement Learning.

    Integrates SUMO/TraCI for simulation control and state extraction.
    Automatically determines intersection neighbors and orders incoming lanes.
    Requires network file (.net.xml) for structural analysis.
    Outputs state observations suitable for multi-agent RL.
    """

    def __init__(
        self,
        sumo_cfg_path,
        net_file_path,
        controlled_intersections,
        max_lanes_per_direction=DEFAULT_MAX_LANES_PER_DIRECTION,
        step_duration=1.0,
        max_simulation_time=3600,
        padding_value=-1.0,
    ):
        """
        Initializes the traffic environment.

        Args:
            sumo_cfg_path (str): Path to the SUMO configuration file (.sumocfg).
            net_file_path (str): Path to the SUMO network file (.net.xml).
            controlled_intersections (list): List of dicts, each defining a controlled intersection.
                                             Requires "id" (SUMO junction ID) and "action_phases" (dict mapping action index to SUMO phase string).
                                             Example: [{"id": "J1", "action_phases": {0: "GrGr", 1: "rGrG"}}]
            max_lanes_per_direction (int): The maximum number of lanes approaching any controlled
                                           intersection from a single cardinal direction (N, E, S, or W).
                                           This is CRITICAL for defining the state vector size.
                                           Defaults to DEFAULT_MAX_LANES_PER_DIRECTION.
            step_duration (float): Duration of one RL step in simulation seconds. Defaults to 1.0.
            max_simulation_time (float): Maximum simulation time (seconds) before episode ends. Defaults to 3600.
            padding_value (float): Value used for padding missing neighbor information. Defaults to -1.0.
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.net_file_path = net_file_path
        self.step_duration = step_duration
        self.max_simulation_time = max_simulation_time
        self.padding_value = padding_value
        self.max_lanes_per_direction = max_lanes_per_direction

        if not os.path.exists(self.net_file_path):
            sys.exit(f"Error: Network file not found: {self.net_file_path}")
        if not controlled_intersections:
            sys.exit("Error: No controlled intersections defined.")

        self.sumo_process = None
        self.current_time = 0.0

        try:
            self.net = sumolib.net.readNet(self.net_file_path)
        except Exception as e:
            sys.exit(f"Error loading SUMO network file '{self.net_file_path}': {e}")

        self.controlled_intersections_config = {
            int_config["id"]: int_config for int_config in controlled_intersections
        }
        self.controlled_intersection_ids = list(
            self.controlled_intersections_config.keys()
        )

        self.traffic_light_ids = self._get_traffic_light_ids()
        self.action_to_sumo_phase = {
            int_id: config.get("action_phases", {})
            for int_id, config in self.controlled_intersections_config.items()
            if int_id in self.traffic_light_ids
        }
        self.controlled_intersection_ids = [
            jid
            for jid in self.controlled_intersection_ids
            if jid in self.traffic_light_ids
        ]
        if not self.controlled_intersection_ids:
            sys.exit(
                "Error: None of the provided 'controlled_intersections' IDs correspond to traffic lights found in the network."
            )

        self.observed_lanes = self._get_ordered_lanes_for_all_intersections()
        self.neighbor_map = self._build_neighbor_map()

        # Local state: (4 directions * max_lanes_per_direction queues) + 2 signal values
        self.local_state_size = (4 * self.max_lanes_per_direction) + 2
        # Neighbor presence vector: 4 elements (N, S, E, W)
        self.presence_vector_size = 4
        # Neighbor state block size (same as local state size)
        self.neighbor_info_size = self.local_state_size
        # Total state size: local_state + presence_vector + 4 * neighbor_state
        self.state_vector_size = (
            self.local_state_size
            + self.presence_vector_size
            + (4 * self.neighbor_info_size)
        )
        self._previous_observations = None
        print(
            f"SumoTrafficEnvironment initialized. State vector size: {self.state_vector_size}"
        )

        # Initialize Keras models for each controlled intersection
        self.models = {}
        for intersection_id in self.controlled_intersection_ids:
            self.models[intersection_id] = self._build_model()

    def _build_model(self):
        """Builds a simple Keras model for RL."""
        model = Sequential([
            Input(shape=(self.state_vector_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.get_action_size(), activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _get_node_by_id(self, node_id):
        """Safely retrieves a sumolib node object by its ID."""
        try:
            return self.net.getNode(node_id)
        except KeyError:
            print(f"Warning: Node ID '{node_id}' not found in the network file.")
            return None

    def _get_lane_index(self, lane_id_str: str) -> int:
        """Extracts the lane index (e.g., _0, _1) from a SUMO lane ID."""
        try:
            parts = lane_id_str.rsplit("_", 1)
            if len(parts) > 1:
                return int(parts[-1])
            return 0
        except (ValueError, IndexError):
            return 0

    def _determine_direction(self, from_node, to_node):
        """Determines the cardinal direction from 'from_node' to 'to_node' based on coordinates."""
        if not from_node or not to_node:
            return None
        try:
            from_coord = from_node.getCoord()
            to_coord = to_node.getCoord()
        except Exception as e:
            # Handle potential errors getting coordinates
            print(
                f"Warning: Could not get coordinates for node {from_node.getID() if from_node else 'None'} or {to_node.getID() if to_node else 'None'}: {e}"
            )
            return None

        dx = to_coord[0] - from_coord[0]
        dy = to_coord[1] - from_coord[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        if 45 <= angle_deg < 135:
            return "N"
        elif 135 <= angle_deg < 225:
            return "W"
        elif 225 <= angle_deg < 315:
            return "S"
        else:
            return "E"

    def _get_traffic_light_ids(self):
        """Identifies junctions controlled by traffic lights using sumolib."""
        traffic_light_junction_ids = {}
        all_nodes = self.net.getNodes()
        for node in all_nodes:
            # Check if the node type itself is traffic_light
            if node.getType() == "traffic_light":
                traffic_light_junction_ids[node.getID()] = (
                    node.getID()
                )  # Store mapping J_ID -> TL_ID (often same)
            # Alternative/Additional check: Does the node *control* a TLS?
            # try:
            #      # This might require specific sumolib versions or methods
            #      if node.getTLS():
            #          # If multiple TLS controlled, logic needed to choose the right one
            #          # For simplicity, assume type check is sufficient or IDs match
            #          if node.getID() not in traffic_light_junction_ids:
            #               # print(f"Info: Junction {node.getID()} controls a TLS but type isn't 'traffic_light'. Adding.")
            #               traffic_light_junction_ids[node.getID()] = node.getID() # Or figure out actual TL ID
            # except AttributeError:
            #      pass # getTLS method might not exist

        # Verify provided controlled IDs against found TL IDs
        valid_tl_ids = {}
        for jid in self.controlled_intersection_ids:
            if jid in traffic_light_junction_ids:
                valid_tl_ids[jid] = traffic_light_junction_ids[jid]
            else:
                print(
                    f"Warning: Provided controlled intersection ID '{jid}' is not identified as a traffic light junction in the network file. It will be ignored."
                )

        return valid_tl_ids

    def _build_neighbor_map(self):
        """Builds map of direct N, E, S, W neighbors for controlled TL junctions using coordinates."""
        neighbor_map = {}
        controlled_node_ids_set = set(self.controlled_intersection_ids)

        for agent_id in self.controlled_intersection_ids:
            agent_node = self._get_node_by_id(agent_id)
            if not agent_node:
                continue

            neighbors = {"N": None, "E": None, "S": None, "W": None}

            # Check nodes connected via outgoing edges
            for edge in agent_node.getOutgoing():
                neighbor_node = edge.getToNode()
                neighbor_id = neighbor_node.getID()
                if neighbor_id in controlled_node_ids_set and neighbor_id != agent_id:
                    direction = self._determine_direction(agent_node, neighbor_node)
                    if direction and neighbors[direction] is None:
                        neighbors[direction] = neighbor_id

            # Check nodes connected via incoming edges (for connections defined other way)
            for edge in agent_node.getIncoming():
                neighbor_node = edge.getFromNode()
                neighbor_id = neighbor_node.getID()
                if neighbor_id in controlled_node_ids_set and neighbor_id != agent_id:
                    reverse_direction = self._determine_direction(
                        neighbor_node, agent_node
                    )
                    direction = None
                    if reverse_direction == "N":
                        direction = "S"
                    elif reverse_direction == "S":
                        direction = "N"
                    elif reverse_direction == "E":
                        direction = "W"
                    elif reverse_direction == "W":
                        direction = "E"

                    if direction and neighbors[direction] is None:
                        neighbors[direction] = neighbor_id
                    # Optional: Add warning if conflict found (neighbor already assigned via outgoing)

            neighbor_map[agent_id] = neighbors
        return neighbor_map

    def _get_ordered_lanes_for_intersection(self, intersection_id):
        """Gets incoming lanes, orders by N, E, S, W approach, sorts by index, pads/truncates."""
        node = self._get_node_by_id(intersection_id)
        # Expected lanes per direction for padding calculation before full list is built
        pad_id_prefix = f"PAD_{intersection_id}"

        if not node:
            return [
                f"{pad_id_prefix}_{i}" for i in range(4 * self.max_lanes_per_direction)
            ]

        lanes_by_direction = {"N": [], "E": [], "S": [], "W": []}
        for edge in node.getIncoming():
            from_node = edge.getFromNode()
            direction = self._determine_direction(from_node, node)
            if direction:
                lanes = edge.getLanes()
                # Sort lanes within the edge based on index before adding
                lanes.sort(key=lambda l: l.getIndex())
                lanes_by_direction[direction].extend(lanes)  # Add sumolib Lane objects

        # Combine lanes in N, E, S, W order and pad/truncate each direction
        final_ordered_lanes = []
        lane_count = 0
        for direction in ["N", "E", "S", "W"]:
            dir_lanes = lanes_by_direction[direction]
            # Sort again just in case (e.g. multiple incoming edges from same direction)
            dir_lanes.sort(key=lambda l: l.getIndex())
            num_lanes_in_dir = len(dir_lanes)

            for i in range(self.max_lanes_per_direction):
                if i < num_lanes_in_dir:
                    final_ordered_lanes.append(dir_lanes[i].getID())  # Add lane ID
                else:
                    # Add padding ID specific to this slot
                    final_ordered_lanes.append(f"{pad_id_prefix}_{direction}_{i}")
                lane_count += 1

        # The final list should inherently have the correct total size
        # final_size = 4 * self.max_lanes_per_direction
        # Add assertion for debugging if needed
        # assert len(final_ordered_lanes) == final_size, f"Lane ordering mismatch for {intersection_id}"

        return final_ordered_lanes

    def _get_ordered_lanes_for_all_intersections(self):
        """Generates the ordered/padded lane lists for all controlled intersections."""
        observed_lanes_map = {}
        for int_id in self.controlled_intersection_ids:
            observed_lanes_map[int_id] = self._get_ordered_lanes_for_intersection(
                int_id
            )
        return observed_lanes_map

    # --- Core Environment Methods ---

    def start_sumo(self, use_gui=False, sumo_seed="random", port=None):
        """Starts the SUMO simulation as a subprocess and connects TraCI."""
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path]
        sumo_cmd.extend(["--seed", str(sumo_seed)])
        # Ensure simulation step length matches TraCI expectation if possible
        # sumo_cmd.extend(["--step-length", str(traci.simulation.getDeltaT())]) # Requires TraCI import early

        if port is None:
            port = sumolib.miscutils.getFreeSocketPort()
        sumo_cmd.extend(["--remote-port", str(port)])

        print(f"Starting SUMO on port {port}: {' '.join(sumo_cmd)}")
        try:
            # Use DEVNULL to hide SUMO output unless debugging
            self.sumo_process = subprocess.Popen(
                sumo_cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr
            )
            time.sleep(2)  # Wait for SUMO to initialize
            traci.connect(port=port)
            print(f"TraCI connected to port {port}.")
            # Ensure simulation is ready
            traci.simulationStep()
            self.current_time = traci.simulation.getTime()

        except traci.exceptions.TraCIException as e:
            print(f"Error connecting TraCI: {e}. Is SUMO running?")
            self._terminate_sumo_process()
            sys.exit(1)
        except FileNotFoundError:
            print(
                f"Error: '{sumo_binary}' command not found. Make sure SUMO is installed and in PATH."
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error starting SUMO or connecting TraCI: {e}")
            self._terminate_sumo_process()
            sys.exit(1)

    def _terminate_sumo_process(self):
        """Safely terminates the SUMO subprocess."""
        if self.sumo_process:
            try:
                if self.sumo_process.poll() is None:
                    self.sumo_process.terminate()
                    self.sumo_process.wait(timeout=5)
                    # print("SUMO process terminated.")
            except subprocess.TimeoutExpired:
                # print("Warning: SUMO process did not terminate gracefully. Forcing kill.")
                self.sumo_process.kill()
                self.sumo_process.wait()
            except Exception as e:
                print(f"Error during SUMO process termination: {e}")
            finally:
                self.sumo_process = None

    def close_sumo(self):
        """Closes the TraCI connection and terminates the SUMO subprocess."""
        if traci.isconnected():
            try:
                traci.close()
                # print("TraCI connection closed.")
            except traci.exceptions.FatalTraCIError:
                pass  # Already closed
            except Exception as e:
                print(f"Error closing TraCI connection: {e}")
        self._terminate_sumo_process()

    def reset(self, sumo_seed="random", use_gui=False):
        """Resets environment, starts SUMO, returns initial observations."""
        # print("\nResetting environment...")
        self.close_sumo()
        time.sleep(0.5)  # Brief pause

        self.start_sumo(use_gui=use_gui, sumo_seed=sumo_seed)
        self.current_time = 0.0  # Reset time after initial step in start_sumo

        initial_observations = self._get_observations()
        self._previous_observations = initial_observations
        # print("Environment reset complete.")
        return initial_observations

    def step(self, actions):
        """Advances the environment by one RL step."""
        if not traci.isconnected():
            print("Error: TraCI is not connected. Cannot step.")
            # Handle appropriately: maybe try reconnecting or return error state
            empty_obs = {
                agent_id: np.full(
                    self.state_vector_size, self.padding_value, dtype=np.float32
                )
                for agent_id in self.controlled_intersection_ids
            }
            empty_rewards = {
                agent_id: 0.0 for agent_id in self.controlled_intersection_ids
            }
            return empty_obs, empty_rewards, True, {"error": "TraCI disconnected"}

        current_observations = (
            self._previous_observations
        )  # Store state before action/step

        self._apply_actions(actions)

        target_time = self.current_time + self.step_duration
        simulation_halted = False
        try:
            while self.current_time < target_time:
                traci.simulationStep()
                self.current_time = traci.simulation.getTime()
                if self._is_episode_done_sumo():
                    simulation_halted = True
                    break
        except traci.exceptions.FatalTraCIError as e:
            print(f"Fatal TraCI error during simulation step: {e}. Ending episode.")
            simulation_halted = True  # Treat connection loss as halt
            # Set done flag later based on this
        except traci.exceptions.TraCIException as e:
            print(
                f"Non-fatal TraCI error during step: {e}"
            )  # Log but continue if possible

        max_time_reached = self.current_time >= self.max_simulation_time
        done = max_time_reached or simulation_halted

        next_observations = self._get_observations()  # Get state *after* stepping
        rewards = self._calculate_rewards(
            current_observations, next_observations, actions
        )
        self._previous_observations = next_observations  # Update for next step

        info = {"simulation_time": self.current_time}
        if simulation_halted:
            info["termination_reason"] = "sumo_halted"
        elif max_time_reached:
            info["termination_reason"] = "max_time_reached"

        return next_observations, rewards, done, info

    def _apply_actions(self, actions):
        """Applies agent actions (selecting traffic light phases) via TraCI."""
        for intersection_id, action_index in actions.items():
            if intersection_id not in self.traffic_light_ids:
                continue

            tl_id = self.traffic_light_ids[intersection_id]
            action_phases = self.action_to_sumo_phase.get(intersection_id, {})
            sumo_phase_string = action_phases.get(action_index)

            if sumo_phase_string:
                try:
                    all_logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                        tl_id
                    )
                    if not all_logics:
                        continue
                    current_logic = all_logics[0]

                    phase_index_in_sumo = -1
                    for i, phase in enumerate(current_logic.phases):
                        if phase.state == sumo_phase_string:
                            phase_index_in_sumo = i
                            break

                    if phase_index_in_sumo != -1:
                        if traci.trafficlight.getPhase(tl_id) != phase_index_in_sumo:
                            traci.trafficlight.setPhase(tl_id, phase_index_in_sumo)
                    else:
                        # This warning is important for debugging phase definitions
                        print(
                            f"Warning: SUMO phase string '{sumo_phase_string}' (Action {action_index}) not found for TL {tl_id}."
                        )

                except traci.exceptions.TraCIException :
                    pass
                except IndexError:
                    print(f"Error accessing phase definitions for TL {tl_id}.")
                except Exception as e:  # Catch other unexpected errors
                    print(f"Unexpected error applying action to TL {tl_id}: {e}")

    def _get_local_observation(self, agent_id):
        """Collects local state (queues, signal) for one agent using TraCI."""
        # Relies on self.observed_lanes being correctly ordered and padded
        ordered_lanes = self.observed_lanes.get(agent_id, [])
        # Expected size based on init parameter
        queue_block_size = 4 * self.max_lanes_per_direction
        local_state_size = queue_block_size + 2
        local_obs = [self.padding_value] * local_state_size  # Initialize with padding

        # 1. Get Queue Counts
        for i, lane_id in enumerate(ordered_lanes):
            if i >= queue_block_size:
                break  # Should not happen if observed_lanes is correct size
            if lane_id.startswith(
                "PAD_"
            ):  # Skip padding lanes defined in _get_ordered_lanes
                local_obs[i] = 0.0  # Use 0 for padded queue slots
                continue
            try:
                # Halting number is generally preferred for queue length in RL
                num_halting = traci.lane.getLastStepHaltingNumber(lane_id)
                local_obs[i] = float(num_halting)
            except traci.exceptions.TraCIException:
                # Keep padding value if TraCI fails for this lane
                pass
            except Exception as e:
                print(f"Unexpected error getting queue for lane {lane_id}: {e}")
                pass  # Keep padding value

        # 2. Get Signal State
        tl_id = self.traffic_light_ids.get(agent_id)
        if tl_id:
            try:
                current_phase_index = float(traci.trafficlight.getPhase(tl_id))
                # Use time-to-next-switch as a feature
                next_switch_time = float(traci.trafficlight.getNextSwitch(tl_id))
                time_to_next_switch = max(0.0, next_switch_time - self.current_time)

                local_obs[queue_block_size] = current_phase_index
                local_obs[queue_block_size + 1] = time_to_next_switch
            except traci.exceptions.TraCIException:
                # Keep padding values if TL state fails
                pass
            except Exception as e:
                print(f"Unexpected error getting TL state for {tl_id}: {e}")
                pass  # Keep padding values
        # else: # No TL ID, keep padding values from initialization

        return local_obs

    def _get_observations(self):
        """Collects the full state observation (local + neighbors) for all agents."""
        observations = {}
        # Cache local observations to avoid redundant TraCI calls
        local_observations_cache = {}
        if not traci.isconnected():  # Check connection before TraCI calls
            print("Warning: TraCI disconnected during observation gathering.")
            # Return padding state for all agents
            for agent_id in self.controlled_intersection_ids:
                observations[agent_id] = np.full(
                    self.state_vector_size, self.padding_value, dtype=np.float32
                )
            return observations

        for agent_id in self.controlled_intersection_ids:
            local_observations_cache[agent_id] = self._get_local_observation(agent_id)

        # Build full state vector including neighbors
        for agent_id in self.controlled_intersection_ids:
            local_data = local_observations_cache[agent_id]
            presence_vector = self._get_neighbor_presence_vector(agent_id)
            neighbor_info_blocks = []
            agent_neighbors = self.neighbor_map.get(agent_id, {})

            for direction in ["N", "E", "S", "W"]:  # Consistent order
                neighbor_id = agent_neighbors.get(direction)
                if neighbor_id and neighbor_id in local_observations_cache:
                    neighbor_local_data = local_observations_cache[neighbor_id]
                    neighbor_info_blocks.extend(neighbor_local_data)
                else:
                    # Add padding if no neighbor or neighbor not controlled/cached
                    neighbor_info_blocks.extend(
                        [self.padding_value] * self.neighbor_info_size
                    )

            state_vector_list = local_data + presence_vector + neighbor_info_blocks
            # Ensure final vector has the expected size, pad if necessary (shouldn't be needed)
            if len(state_vector_list) != self.state_vector_size:
                print(
                    f"Warning: State vector size mismatch for {agent_id}. Expected {self.state_vector_size}, got {len(state_vector_list)}. Padding/truncating."
                )
                state_vector_list.extend(
                    [self.padding_value]
                    * (self.state_vector_size - len(state_vector_list))
                )
                state_vector_list = state_vector_list[: self.state_vector_size]

            observations[agent_id] = np.array(state_vector_list, dtype=np.float32)

        return observations

    def _get_neighbor_presence_vector(self, agent_id):
        """Generates 4-element vector [N, E, S, W] indicating neighbor presence using neighbor_map."""
        presence_vector = [0.0] * 4
        agent_neighbors = self.neighbor_map.get(agent_id, {})
        if agent_neighbors.get("N"):
            presence_vector[0] = 1.0
        if agent_neighbors.get("E"):
            presence_vector[1] = 1.0
        if agent_neighbors.get("S"):
            presence_vector[2] = 1.0
        if agent_neighbors.get("W"):
            presence_vector[3] = 1.0
        return presence_vector

    def _get_neighbor_id_in_direction(self, agent_id, direction):
        """Gets the ID of the neighbor in the specified direction using neighbor_map."""
        agent_neighbors = self.neighbor_map.get(agent_id, {})
        return agent_neighbors.get(direction)  # Returns None if no neighbor

    def _calculate_rewards(self, previous_states, current_states, actions):
        """Calculates reward based on reduction in local queue lengths."""
        rewards = {}
        if previous_states is None:
            return {agent_id: 0.0 for agent_id in self.controlled_intersection_ids}

        queue_block_size = (
            4 * self.max_lanes_per_direction
        )  # Size of queue part of local state

        for agent_id in self.controlled_intersection_ids:
            prev_state_np = previous_states.get(agent_id)
            curr_state_np = current_states.get(agent_id)

            if prev_state_np is None or curr_state_np is None:
                rewards[agent_id] = 0.0
                continue

            # Extract only the queue lengths (first part of the local state)
            # Handle padding values (-1.0 or 0.0) by treating them as 0 queue length for reward calculation
            prev_queues = np.maximum(0, prev_state_np[:queue_block_size])
            curr_queues = np.maximum(0, curr_state_np[:queue_block_size])

            # Reward = reduction in sum of local queues
            reward = np.sum(prev_queues) - np.sum(curr_queues)
            rewards[agent_id] = reward

        return rewards

    def _is_episode_done_sumo(self):
        """Checks if SUMO simulation has ended (no vehicles left)."""
        if not traci.isconnected():
            return True
        try:
            # Check if vehicles are running OR expected to run
            return (
                traci.simulation.getMinExpectedNumber() <= 0
                and traci.vehicle.getIDCount() <= 0
            )
        except traci.exceptions.TraCIException:
            return True  # Assume done if TraCI fails

    def get_controlled_intersection_ids(self):
        """Returns list of IDs for controlled traffic light intersections."""
        return list(self.controlled_intersection_ids)  # Return a copy

    def get_state_size(self):
        """Returns the calculated size of the state vector for one agent."""
        return self.state_vector_size

    def get_action_size(self, intersection_id=None):
        """Returns the number of actions defined for an agent."""
        # If no specific ID given, return for the first agent (assuming homogeneity)
        target_id = (
            intersection_id
            if intersection_id in self.action_to_sumo_phase
            else (
                self.controlled_intersection_ids[0]
                if self.controlled_intersection_ids
                else None
            )
        )
        if target_id:
            return len(self.action_to_sumo_phase.get(target_id, {}))
        return 0

    def train_agents(self, episodes=100, gamma=0.99):
        """Trains the agents using a simple Q-learning approach."""
        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")
            states = self.reset()
            done = False

            while not done:
                actions = {}
                for agent_id, state in states.items():
                    q_values = self.models[agent_id].predict(state.reshape(1, -1))
                    actions[agent_id] = np.argmax(q_values)

                next_states, rewards, done, _ = self.step(actions)

                for agent_id in self.controlled_intersection_ids:
                    target = rewards[agent_id]
                    if not done:
                        next_q_values = self.models[agent_id].predict(next_states[agent_id].reshape(1, -1))
                        target += gamma * np.max(next_q_values)

                    q_values = self.models[agent_id].predict(states[agent_id].reshape(1, -1))
                    q_values[0][actions[agent_id]] = target

                    self.models[agent_id].fit(states[agent_id].reshape(1, -1), q_values, verbose=0)

                states = next_states

        print("Training complete.")


if __name__ == "__main__":
    try:
        import tensorflow as tf

        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print("TensorFlow not found. Skipping TF-specific parts of example.")

    # --- Configuration ---
    # --- Generic Placeholder - **UPDATE THESE** ---
    dummy_cfg_path = "dummy_scenario.sumocfg"
    dummy_net_path = "dummy_scenario.net.xml"
    # Create dummy files if they don't exist for basic script execution
    if not os.path.exists(dummy_cfg_path):
        with open(dummy_cfg_path, "w") as f:
            f.write(
                f'<configuration><input><net-file value="{os.path.basename(dummy_net_path)}"/></input><time><begin value="0"/></time></configuration>'
            )
        print(f"Created dummy config file: {dummy_cfg_path}")
    if not os.path.exists(dummy_net_path):
        with open(dummy_net_path, "w") as f:
            f.write(
                '<net version="1.1"><location netOffset="0,0" convBoundary="0,0,100,100" origBoundary="-1000,-1000,1000,1000" projParameter="!"/><junction id="J_dummy" type="traffic_light" x="50" y="50" incLanes="" intLanes="" shape="50,55 55,50 50,45 45,50"/><tlLogic id="J_dummy" type="static" programID="0" offset="0"><phase duration="30" state="Gr"/><phase duration="5" state="yr"/><phase duration="30" state="rG"/><phase duration="5" state="ry"/></tlLogic></net>'
            )
        print(f"Created dummy network file: {dummy_net_path}")

    YOUR_SUMO_CFG_FILE = dummy_cfg_path  # <-- **UPDATE THIS PATH**
    YOUR_NET_FILE = dummy_net_path  # <-- **UPDATE THIS PATH**

    # Define controlled intersections and their action->phase mapping
    CONTROLLED_INTERSECTIONS = [
        {
            "id": "J_dummy",  # Must exist in the .net.xml and be type="traffic_light"
            "action_phases": {
                0: "Gr",  # Action 0 selects phase "Gr" (must be defined in tlLogic)
                1: "rG",  # Action 1 selects phase "rG" (must be defined in tlLogic)
            },
        },
    ]
    # **IMPORTANT**: Define maximum lanes approaching from one direction (N, E, S, or W)
    # This MUST match your network's characteristics for correct state size.
    # For the dummy network, it's 0, but use a realistic value for real nets.
    MAX_LANES_PER_DIR_CONFIG = 1  # <-- **UPDATE FOR YOUR NETWORK**
    # --- End Generic Placeholder ---

    if not os.path.exists(YOUR_SUMO_CFG_FILE):
        sys.exit(f"Error: SUMO cfg not found: {YOUR_SUMO_CFG_FILE}")
    if not os.path.exists(YOUR_NET_FILE):
        sys.exit(f"Error: SUMO net not found: {YOUR_NET_FILE}")
    # --- End Configuration ---

    # 1. Initialize
    try:
        env = SumoTrafficEnvironment(
            sumo_cfg_path=YOUR_SUMO_CFG_FILE,
            net_file_path=YOUR_NET_FILE,
            controlled_intersections=CONTROLLED_INTERSECTIONS,
            max_lanes_per_direction=MAX_LANES_PER_DIR_CONFIG,
            step_duration=5.0,
            max_simulation_time=60,  # Short duration for example
        )
    except SystemExit as e:
        print(f"Failed to initialize environment: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during init: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 2. Reset
    try:
        initial_states_np = env.reset(use_gui=False)  # Set use_gui=True to watch
    except SystemExit:
        print("Failed to reset environment (SUMO start failed?). Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during reset: {e}")
        import traceback

        traceback.print_exc()
        env.close_sumo()
        sys.exit(1)

    print(f"\nInitial State Shapes (Agent: Shape):")
    for agent_id, state_np in initial_states_np.items():
        print(f"  {agent_id}: {state_np.shape}")
        if TF_AVAILABLE:
            state_tensor = tf.expand_dims(
                tf.convert_to_tensor(state_np, dtype=tf.float32), 0
            )
            print(f"    TF Tensor Shape: {state_tensor.shape}")

    # 3. Simulation Loop
    done = False
    step_count = 0
    current_states_np = initial_states_np

    print("\nStarting simulation loop...")
    while not done:
        step_count += 1
        actions = {}
        agent_ids = env.get_controlled_intersection_ids()
        if not agent_ids:
            print("Error: No controllable agents available.")
            break

        for agent_id in agent_ids:
            action_size = env.get_action_size(agent_id)
            if action_size > 0:
                actions[agent_id] = np.random.randint(
                    0, action_size
                )  # Random action selection
            # else: print(f"Warning: Agent {agent_id} has action size 0.")

        if not actions:
            print("No actions generated. Stopping.")
            break

        try:
            next_states_np, rewards, done, info = env.step(actions)
        except Exception as e:
            print(f"\n--- Critical Error during env.step() at step {step_count} ---")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            done = True  # Force stop
            rewards = {agent_id: 0.0 for agent_id in actions.keys()}
            next_states_np = current_states_np
            info = {"error": str(e)}

        current_states_np = next_states_np
        sim_time = info.get("simulation_time", env.current_time)
        term_reason = info.get("termination_reason", "")
        err_info = info.get("error", "")
        print(
            f"Step {step_count}: SimTime {sim_time:.2f}, Actions: {actions}, Rewards: {rewards}, Done: {done} {term_reason} {err_info}"
        )

    print("\nEpisode finished.")

    # 4. Close
    env.close_sumo()
    print("Environment closed.")

    # Clean up dummy files
    if (
        "dummy_cfg_path" in locals()
        and os.path.basename(YOUR_SUMO_CFG_FILE) == os.path.basename(dummy_cfg_path)
        and os.path.exists(dummy_cfg_path)
    ):
        os.remove(dummy_cfg_path)
        print(f"Removed dummy config file: {dummy_cfg_path}")
    if (
        "dummy_net_path" in locals()
        and os.path.basename(YOUR_NET_FILE) == os.path.basename(dummy_net_path)
        and os.path.exists(dummy_net_path)
    ):
        os.remove(dummy_net_path)
        print(f"Removed dummy network file: {dummy_net_path}")

    # Train the agents
    env.train_agents(episodes=10)
