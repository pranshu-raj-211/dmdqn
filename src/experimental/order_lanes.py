import traci
import sumolib.net
import os
import sys
import subprocess
import time
import numpy as np
from log_config import logger

DIRECTION_ORDER = {"n": 0, "s": 1, "e": 2, "w": 3, "unknown": 4, "error": 5}
SUMO_CFG_PATH = "src/sumo_files/scenarios/grid_3x3_lefthand/grid_3x3_lht.sumocfg"
SUMO_NET_PATH = "src/sumo_files/scenarios/grid_3x3_lefthand/grid_3x3_lht.net.xml"
PORT = 8813
PHASE_ENCODING = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    if not os.path.exists(tools):
        print(
            f"Error: SUMO tools directory not found at '{tools}'. Please verify SUMO_HOME."
        )
else:
    print("SUMO_HOME environment variable not found.")
    default_sumo_home = "/usr/share/sumo"
    if os.path.exists(default_sumo_home):
        os.environ["SUMO_HOME"] = default_sumo_home
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
        if not os.path.exists(tools):
            print(
                f"Error: SUMO tools directory not found at '{tools}' after attempting default. Please verify your SUMO installation and SUMO_HOME variable."
            )
    else:
        print(
            "Could not find a default SUMO_HOME path. Please set the SUMO_HOME environment variable."
        )


def get_approach_direction_from_id(lane_id):
    """
    Infers the compass direction of an incoming edge's approach based on the edge ID string pattern.

    Assumes edge IDs follow patterns like:
    - 'END_[Direction]_[row]_[col]_to_J_[row]_[col]' (e.g., 'END_N_0_0_to_J_0_0')
    - 'J_[from_row]_[from_col]_to_J_[to_row]_[to_col]' (e.g., 'J_0_0_to_J_0_1')
    - 'J_[from_row]_[from_col]_to_END_[Direction]_[row]_[col]' (These are outgoing, should be filtered earlier)

    Args:
        edge_id (str): The ID of the incoming edge.

    Returns:
        str: Lowercase single letter representing the approach direction ('n', 'e', 's', 'w'),
             or 'unknown'/'error' if the pattern is not recognized.
    """
    try:
        parts = lane_id.split("_to_")
        if len(parts) != 2:
            return "error", None

        from_part = parts[0]
        to_part = parts[1]

        # Case 1: Edge comes from an END node (e.g., 'END_N_0_0_to_J_0_0')
        if from_part.startswith("END_"):
            end_direction_part = from_part.split("_")[1]  # 'N', 'S', 'E', 'W'
            if end_direction_part in {"N", "S", "E", "W"}:
                return end_direction_part.lower(), to_part
            else:
                return "unknown", None  # Unrecognized END direction

        # Case 2: Edge comes from another Junction (e.g., 'J_0_0_to_J_0_1')
        elif from_part.startswith("J_") and to_part.startswith("J_"):
            from_parts = from_part.split("_")
            to_parts = to_part.split("_")
            from_row = int(from_parts[1])
            from_col = int(from_parts[2])
            to_row = int(to_parts[1])
            to_col = int(to_parts[2])

            # Determine edge direction based on index change
            if from_row < to_row:  # Row index increases -> from north
                return "n", to_part
            elif from_row > to_row:  # Row index decreases -> from south
                return "s", to_part
            elif from_col < to_col:  # Column index increases -> from west
                return "w", to_part
            elif from_col > to_col:  # Column index decreases -> from east
                return "e", to_part
            else:
                return "unknown", None
        else:
            # Unrecognized 'from' part pattern
            return "unknown", None

    except Exception as e:
        print(f"Error parsing lane ID '{lane_id}': {e}")
        return "error", None


def get_traffic_light_junction_ids_and_net_sumolib(net_file_path):
    """
    Reads the network file using sumolib and returns IDs of traffic light junctions
    and the sumolib network object.
    """
    traffic_light_ids = []
    net = None
    print(f"\nReading network file '{net_file_path}' using sumolib.")
    try:
        if not os.path.exists(net_file_path):
            print(f"Error: Network file not found at '{net_file_path}'.")
            return traffic_light_ids, net

        net = sumolib.net.readNet(net_file_path)
        print("Network file read successfully by sumolib.")
        all_nodes = net.getNodes()
        print(f"Found {len(all_nodes)} nodes/junctions in the .net.xml file.")

        print("Filtering for traffic light junctions using sumolib.")
        for node in all_nodes:
            if node.getType() == "traffic_light":
                traffic_light_ids.append(node.getID())

        print(f"Found {len(traffic_light_ids)} traffic light junctions via sumolib.")
        return traffic_light_ids

    except Exception as e:
        print(f"\nError reading network file with sumolib: {e}")
        print(
            f"Please ensure '{net_file_path}' is a valid SUMO network file and sumolib is installed/accessible."
        )
    return traffic_light_ids, net


def build_junction_lane_mapping(junction_ids: list, lane_ids: list) -> dict:
    junction_lane_map = {junction_id: [[], [], [], []] for junction_id in junction_ids}
    try:
        for lane_id in lane_ids:
            direction, junction = get_approach_direction_from_id(lane_id)
            if direction == "unknown":
                continue
            junction_lane_map[junction[:5]][DIRECTION_ORDER[direction]].append(lane_id)
    except Exception as e:
        print("Exception occured while building mapping", e)
        print(junction_lane_map)
        print(lane_id, direction, junction)
    return junction_lane_map


def start_sumo(use_gui=False, sumo_seed="42", port=PORT):
    """Starts the SUMO simulation as a subprocess and connects TraCI."""
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    # Check if the binary exists and is executable
    try:
        # Use shell=True on Windows if the binary is just a name like "sumo-gui.exe"
        # Otherwise, prefer shell=False for better security and error handling
        subprocess.run(
            [sumo_binary, "-V"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=(sys.platform == "win32"),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"\nError: SUMO binary '{sumo_binary}' not found or not executable. Details: {e}"
        )
        print("Please ensure SUMO is installed and accessible in your system's PATH.")
        if "SUMO_HOME" in os.environ:
            print(
                f"Also check that '{os.path.join(os.environ.get('SUMO_HOME'), 'bin', sumo_binary)}' exists."
            )
        sys.exit(1)

    # Build the SUMO command to be run as a subprocess
    sumo_cmd = [sumo_binary, "-c", SUMO_CFG_PATH]
    sumo_cmd.extend(["--seed", str(sumo_seed)])
    sumo_cmd.extend(["--step-length", "0.1"])
    sumo_cmd.extend(["--time-to-teleport", "-1"])  # Disable teleporting vehicles
    # Specify the remote port for TraCI connection
    sumo_cmd.extend(["--remote-port", str(port)])

    print(f"\nStarting SUMO process: {' '.join(sumo_cmd)}")
    logger.debug("Starting SUMO process")
    sumo_process = None

    try:
        # Start SUMO as a separate process
        sumo_process = subprocess.Popen(
            sumo_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait a moment for SUMO to start and open the port
        time.sleep(2)  # Increased wait time slightly

        # Connect TraCI to the running SUMO process on the specified port
        print(f"Attempting to connect TraCI on port {port}...")
        logger.debug("Attempting to connect TraCI on port")
        # Use a loop for connection retries as SUMO might take a moment to be ready
        retry_count = 0
        max_retries = 15  # Increased retries slightly
        while retry_count < max_retries:
            try:
                print(f"Trying to connect to traci: {retry_count} tried")
                traci.init(port=port)
                print("TraCI connection successful.")
                logger.debug("TraCI connection successful")
                break  # Exit retry loop on success
            except ConnectionRefusedError:
                retry_count += 1
                print(
                    f"Connection refused. Retrying in 1 second... (Attempt {retry_count}/{max_retries})"
                )
                if sumo_process and sumo_process.poll() is not None:
                    print("\n--- SUMO Process Died During Connection Attempt ---")
                    stdout, stderr = sumo_process.communicate()
                    print(f"SUMO process exited with code {sumo_process.returncode}")
                    if stdout:
                        print("SUMO STDOUT:\n", stdout.decode(errors="ignore"))
                    if stderr:
                        print("SUMO STDERR:\n", stderr.decode(errors="ignore"))
                    print("---------------------------------------------------")
                time.sleep(1)
            except Exception as e:
                print(
                    f"An unexpected error occurred during TraCI connection attempt: {e}"
                )
                # Don't retry for unexpected errors
                if sumo_process and sumo_process.poll() is None:
                    print("\n--- SUMO Process Died During Connection Attempt ---")
                    stdout, stderr = sumo_process.communicate()
                    print(f"SUMO process exited with code {sumo_process.returncode}")
                    if stdout:
                        print("SUMO STDOUT:\n", stdout.decode(errors="ignore"))
                    if stderr:
                        print("SUMO STDERR:\n", stderr.decode(errors="ignore"))
                    print("---------------------------------------------------")
                    sys.exit(1)
                    sumo_process.terminate()
                    try:
                        sumo_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        sumo_process.kill()
                        sumo_process.wait()
                sys.exit(1)

        if retry_count == max_retries:
            print("\n--- Connection Error ---")
            print(
                f"Failed to connect to TraCI on port {port} after {max_retries} attempts."
            )
            print("Is SUMO running and listening on the specified port?")
            print("Check SUMO's console output for errors.")
            print("------------------------")
            if sumo_process and sumo_process.poll() is None:
                sumo_process.terminate()
                sumo_process.wait()
            sys.exit(1)

        # After connecting, ensure simulation is loaded by performing a first step
        # This also confirms the connection is fully functional
        try:
            traci.simulationStep()
            print("Performed initial simulation step to confirm connection.")
            sim_time = traci.simulation.getTime()
            print(f"Current simulation time after initial step: {sim_time:.1f}s")
        except traci.exceptions.TraCIException as e:
            print(f"Error during initial simulation step after TraCI init: {e}")
            print(
                "This might indicate an issue with the simulation file or connection loading."
            )
            # Close connection and terminate process if the very first step fails
            if traci.isLoaded():
                traci.close()
            if sumo_process and sumo_process.poll() is None:
                sumo_process.terminate()
                try:
                    sumo_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sumo_process.kill()
                    sumo_process.wait()
                sys.exit(1)

        if not traci.isLoaded():
            raise ConnectionError(
                "TraCI connection initialized but simulation not loaded."
            )

        delta_t = traci.simulation.getDeltaT()
        print(f"TraCI connected and simulation loaded. Simulation deltaT: {delta_t}s")
        return sumo_process
    except traci.exceptions.TraCIException as e:
        print("\n--- TraCI Error during init/load ---")
        print(f"A TraCI specific error occurred: {e}")
        print(
            f"Please ensure the SUMO config file '{SUMO_CFG_PATH}' is valid and the network can be loaded."
        )
        print(
            "This error often indicates a problem with loading the network or routes after connection."
        )
        print("---------------------")
        if sumo_process and sumo_process.poll() is None:
            sumo_process.terminate()
            sumo_process.wait()
            if traci.isLoaded():
                traci.close()
            sys.exit(1)
    except Exception as e:
        print("\n--- General Error starting SUMO or connecting TraCI ---")
        print(e)
        print(f"Please ensure the config file exists: {SUMO_CFG_PATH}")
        print(f"Also check if another process is already using port {port}.")
        print("----------------------------------------------------")
        if sumo_process and sumo_process.poll() is None:
            sumo_process.terminate()
            sumo_process.wait()
        if traci.isLoaded():
            traci.close()
        sys.exit(1)


def close_sumo():
    if traci.isconnected():
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass
        except Exception as e:
            print(f"Error closing TraCI connection: {e}")
    # TODO: how to terminate the sumo process? do this to hard reset sumo after each episode


def reset_sumo(sumo_seed="42", use_gui=False):
    close_sumo()
    time.sleep(1)
    process = start_sumo(use_gui=use_gui, sumo_seed=sumo_seed)

    current_time = 0.0
    global_state = dict()
    for junction in tl_junctions:
        global_state[junction] = get_own_state(
            junction_id=junction,
            structured_junction_lane_map=ordered_junction_lane_map,
            max_lanes_per_direction=3,
            current_sim_time=sim_time,
        )
    return process, current_time, global_state


def _parse_junction_id(junction_id_str: str) -> tuple[int, int] | None:
    """Returns row and column - coordinates in matrix of the current junction."""
    if not junction_id_str or not junction_id_str.startswith("J_"):
        return None
    parts = junction_id_str.split("_")
    if len(parts) != 3:
        return None
    try:
        row = int(parts[1])
        col = int(parts[2])
        return row, col
    except ValueError:
        return None


def get_lane_index(lane_id_str: str) -> int:
    """Which part of edge the current lane is (right side driving).

    0: Right
    1: Middle
    2: Left

    Used for ordering, right to left."""
    try:
        parts = lane_id_str.rsplit("_", 1)
        if len(parts) > 1:
            return int(parts[-1])
        return 0
    except (ValueError, IndexError):
        return 0


def _get_neighbor_info(
    current_junction_id: str, tl_junctions: list[str]
) -> tuple[list[int], list[str | None]]:
    """Returns a vector detailing presence of neighbors, list of their ids if present."""
    presence_vector = [0] * 4
    neighbor_id_list = [None] * 4

    neighbor_deltas = {
        "n": (-1, 0),
        "s": (+1, 0),
        "e": (0, +1),
        "w": (0, -1),
    }
    neighbor_indices_in_vector = {"n": 0, "s": 1, "e": 2, "w": 3}

    current_coords = _parse_junction_id(current_junction_id)
    # junction must be a valid traffic light junction
    if current_coords is None:
        return presence_vector, neighbor_id_list

    current_row, current_col = current_coords

    # find actual neighbor if exists
    for direction_str in ["n", "s", "e", "w"]:
        delta = neighbor_deltas[direction_str]
        neighbor_vec_index = neighbor_indices_in_vector[direction_str]

        neighbor_row = current_row + delta[0]
        neighbor_col = current_col + delta[1]
        potential_neighbor_id = f"J_{neighbor_row}_{neighbor_col}"

        if potential_neighbor_id in tl_junctions:
            presence_vector[neighbor_vec_index] = 1
            neighbor_id_list[neighbor_vec_index] = potential_neighbor_id

    return presence_vector, neighbor_id_list


def get_own_state(
    junction_id: str,
    structured_junction_lane_map: dict[str, list[list[str]]],
    max_lanes_per_direction: int,
    current_sim_time: float,
) -> list[float]:
    """Get state of a single junction.
    
    Returns list of size 9 (4 edge blockage + 4 phase + 1 time spent in phase)."""
    # block_size = 4  # using one integer to represent whole edge

    state_block = list()

    if junction_id not in structured_junction_lane_map:
        return [-1.0] * 9

    directional_lists = structured_junction_lane_map[junction_id]

    for dir_list_index in range(len(directional_lists)):
        lanes_list = directional_lists[dir_list_index]
        queue_length = 0

        for lane_id in lanes_list:
            try:
                queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
                # need lane id to detector id mapping for this to work
                # lane_length_covered = traci.lanearea.getJamLengthMeters()
            except traci.exceptions.TraCIException:
                pass
            except Exception:
                pass
        state_block.append(float(queue_length))

    phase = [0, 0, 0, 0]
    time_spent = -1.0

    try:
        if traci.junction.getType(junction_id) == "traffic_light":
            current_phase = traci.trafficlight.getPhase(junction_id)
            # one hot encode phase, zeros for padding
            phase = PHASE_ENCODING.get(current_phase, [0, 0, 0, 0])

            try:
                next_switch_time = traci.trafficlight.getNextSwitch(junction_id)
                current_phase_duration = traci.trafficlight.getPhaseDuration(
                    junction_id
                )
                time_spent_calc = current_phase_duration - (
                    next_switch_time - current_sim_time
                )
                assert time_spent_calc >= 0, "Time spent cannot be negative"
                time_spent = max(0.0, time_spent_calc)
            except traci.exceptions.TraCIException:
                pass
            except Exception:
                pass

    # !exceptions should not be allowed to pass silently
    except traci.exceptions.TraCIException:
        pass
    except Exception:
        pass

    state_block.extend(phase)
    state_block.append(time_spent)

    assert len(state_block) == 9, "Local state size unexpected"

    return state_block


def build_state_vector(
    junction_id: str,
    tl_junctions: list[str],
    structured_junction_lane_map: dict[str, list[list[str]]],
    max_lanes_per_direction: int,
    current_sim_time: float,
    global_state: dict[str, list[float]],
) -> list[float]:
    """Build the observation for the agent at any particular junction.

    This includes local state, neighbor presence vector, neighbor local states.
    """
    max_lanes_per_direction = 1
    block_size = (4 * max_lanes_per_direction) + 5
    padding_block = [-1.0] * block_size

    # junction validity check
    if (
        junction_id not in tl_junctions
        or junction_id not in structured_junction_lane_map
        or _parse_junction_id(junction_id) is None
    ):
        return [-1.0] * (block_size + 4 + (4 * block_size))

    state_vector = get_own_state(
        junction_id,
        structured_junction_lane_map,
        max_lanes_per_direction,
        current_sim_time,
    )

    presence_vector, neighbor_id_list = _get_neighbor_info(junction_id, tl_junctions)

    state_vector.extend(presence_vector)

    # logger.debug(f"Building state vector for junction {junction_id}")
    # logger.debug(f"Own state size: {len(own_state)}")
    # logger.debug(f"Presence vector: {presence_vector}")
    # for neighbor_id in neighbor_id_list:
    #     if neighbor_id is not None and neighbor_id in global_state:
    #         logger.debug(f"Neighbor {neighbor_id} state size: {len(global_state[neighbor_id])}")
    #     else:
    #         logger.debug(f"Neighbor {neighbor_id} is missing or padded.")

    for neighbor_id in neighbor_id_list:
        if neighbor_id is not None and neighbor_id in global_state:
            # TODO: remove lane facing the main junction (not needed, not incoming to current)
            nbr_state = global_state[neighbor_id]
        else:
            nbr_state = padding_block
        state_vector.extend(nbr_state)

    logger.debug(f"{junction_id}: state vector length {len(state_vector)}")
    assert (
        len(state_vector) == 49
    ), f"Unexpected state vector length {len(state_vector)}"
    return np.array(state_vector)


def order_lanes_in_edge(junction_lane_map: dict[str, list[list[str]]]):
    for junction, directions in junction_lane_map.items():
        for index, direction in enumerate(directions):
            direction.sort(key=get_lane_index)
    return junction_lane_map


def calculate_neighbors(junction_id: str) -> list[int]:
    """Builds the presence vector for a junction in a 3x3 grid given its id(location in grid)."""
    x, y = junction_id.split("_")[1], junction_id.split("_")[2]
    presence_vector = [1, 1, 1, 1]
    if x == 0:
        presence_vector[3] = 0
    if x == 2:
        presence_vector[2] = 0
    if y == 0:
        presence_vector[0] = 0
    if y == 2:
        presence_vector[1] = 0
    return presence_vector


def get_full_junction_state(
    junction_id: str,
    tl_junctions: list[str],
    structured_junction_lane_map: dict[str, list[list[str]]],
    max_lanes_per_direction: int,
) -> list[float]:
    """
    Gets the full state vector for a given junction, including its state
    and the states of its adjacent traffic light neighbors. Fetches the
    current simulation time from TraCI internally.

    Args:
        junction_id: The ID of the junction for which to build the state.
        tl_junctions: A list of all traffic light junction IDs in the network.
        structured_junction_lane_map: A dict {jid: [[N_lanes],[S_lanes],[E_lanes],[W_lanes]], ...}
                                      where inner lists are sorted by lane index,
                                      and the order of the inner lists is N, S, E, W.
        max_lanes_per_direction: The maximum number of lanes any single incoming
                                 approach direction has in the network (e.g., 3).

    Returns:
        A list of floats representing the padded state vector (89 elements if max_lanes=3).
        Returns a vector initialized with padding (-1.0) if the current_junction_id
        is invalid, not a TL, or not in the lane map.
    """
    # Get the current simulation time from TraCI
    try:
        current_sim_time = traci.simulation.getTime()
    except traci.exceptions.TraCIException as e:
        print(f"Error getting current simulation time from TraCI: {e}")
        # Return a padding vector if simulation time cannot be retrieved
        block_size = (4 * max_lanes_per_direction) + 2
        return [-1.0] * (block_size + 4 + (4 * block_size))
    except Exception as e:
        print(f"An unexpected error occurred getting simulation time: {e}")
        block_size = (4 * max_lanes_per_direction) + 2
        return [-1.0] * (block_size + 4 + (4 * block_size))

    state_vector = build_state_vector(
        junction_id,
        tl_junctions,
        structured_junction_lane_map,
        max_lanes_per_direction,
        current_sim_time,
        global_state,
    )

    return state_vector


if __name__ == "__main__":
    start_sumo(use_gui=False)
    all_lanes = traci.lane.getIDList()
    non_internal_lanes = [lane for lane in all_lanes if not lane.startswith(":")]
    print("len non internal lanes", len(non_internal_lanes))

    tl_junctions = get_traffic_light_junction_ids_and_net_sumolib(SUMO_NET_PATH)

    junction_lane_map = build_junction_lane_mapping(tl_junctions, non_internal_lanes)
    ordered_junction_lane_map = order_lanes_in_edge(junction_lane_map)

    sim_time = traci.simulation.getTime()

    global_state = dict()
    for junction in tl_junctions:
        global_state[junction] = get_own_state(
            junction_id=junction,
            structured_junction_lane_map=ordered_junction_lane_map,
            max_lanes_per_direction=3,
            current_sim_time=sim_time,
        )

    state = build_state_vector(
        junction_id="J_0_0",
        tl_junctions=tl_junctions,
        structured_junction_lane_map=ordered_junction_lane_map,
        max_lanes_per_direction=3,
        current_sim_time=sim_time,
        global_state=global_state,
    )
    print(state)
    state = build_state_vector(
        junction_id="J_1_1",
        tl_junctions=tl_junctions,
        structured_junction_lane_map=ordered_junction_lane_map,
        max_lanes_per_direction=3,
        current_sim_time=sim_time,
        global_state=global_state,
    )
    print(state)
