import traci
import sumolib.net
import os
import sys
import subprocess
import time


DIRECTION_ORDER = {"n": 0, "s": 1, "e": 2, "w": 3, "unknown": 4, "error": 5}
SUMO_CFG_PATH = "src/sumo_files/scenarios/grid_3x3.sumocfg"
SUMO_NET_PATH = "src/sumo_files/scenarios/grid_3x3.net.xml"
PORT = 8813


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


def get_ordered_incoming_lanes_by_junction(
    traffic_light_junction_ids: list, lane_ids: list
):
    """
    Generates a dictionary mapping each traffic light junction ID to a list of its
    incoming network lane IDs, ordered first by approach direction (NSEW, inferred
    from ID) and then by lane index (0, 1, 2...).

    Args:
        traffic_light_junction_ids (list): A list of IDs of traffic light junctions.

    Returns:
        dict: A dictionary {junction_id: [ordered_incoming_lane_ids]}.
              Returns an empty list for a junction if no incoming network edges/lanes are found.
    """
    junction_lane_map = {}

    print("\n--- Ordering Incoming Lanes by Junction (NSEW + Lane Index, ID Based) ---")

    for junction_id in traffic_light_junction_ids:
        try:
            mixed_incoming_ids = traci.junction.getIncomingEdges(junction_id)

            # Filter the list to keep only the actual network edge IDs
            # Network edge IDs typically do not start with a colon ':'
            incoming_network_edge_ids = [
                id for id in mixed_incoming_ids if not id.startswith(":")
            ]

            # List to store lane IDs for this junction, with sorting information
            lanes_with_sorting_info = []

            if incoming_network_edge_ids:
                # Create a list of tuples (direction, edge_id) for sorting edges
                edges_with_direction = []
                for edge_id in incoming_network_edge_ids:
                    # Use the new ID-based direction inference function
                    direction = get_approach_direction_from_id(edge_id, junction_id)
                    edges_with_direction.append((direction, edge_id))

                # Sort edges by direction (NSEW)
                sorted_edges = sorted(
                    edges_with_direction,
                    key=lambda item: DIRECTION_ORDER.get(item[0], 5),
                )

                # Now, for each sorted edge, get and sort its lanes
                for direction, edge_id in sorted_edges:
                    try:
                        lane_ids = traci.edge.getLaneIds(edge_id)

                        # Sort lanes by their index (the number after the last underscore)
                        # Example: 'edgeID_0', 'edgeID_1', 'edgeID_2'
                        def get_lane_index(lane_id):
                            try:
                                index_str = lane_id.split("_")[-1]
                                return int(index_str)
                            except (ValueError, IndexError):
                                # Return a high number if index cannot be parsed, to sort them last
                                return 999

                        sorted_lane_ids = sorted(lane_ids, key=get_lane_index)
                        lanes_with_sorting_info.extend(sorted_lane_ids)

                    except traci.exceptions.TraCIException as e:
                        print(
                            f"Error getting lanes for edge '{edge_id}' for junction '{junction_id}': {e}"
                        )
                    except Exception as e:
                        print(
                            f"General error getting lanes for edge '{edge_id}' for junction '{junction_id}': {e}"
                        )

            junction_lane_map[junction_id] = lanes_with_sorting_info

            print(f"  Junction '{junction_id}': Ordered incoming lanes:")
            if lanes_with_sorting_info:
                for lane_id in lanes_with_sorting_info:
                    print(f"    - {lane_id}")
            else:
                print("    (No incoming network lanes found or processed)")

        except traci.exceptions.TraCIException as e:
            print(f"TraCI error processing junction '{junction_id}': {e}")
            return None
        except Exception as e:
            print(f"General error processing junction '{junction_id}': {e}")
            return None

    print("--- Ordering Complete ---")
    return junction_lane_map


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
    sumo_process = None

    try:
        # Start SUMO as a separate process
        sumo_process = subprocess.Popen(
            sumo_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a moment for SUMO to start and open the port
        time.sleep(2)  # Increased wait time slightly

        # Connect TraCI to the running SUMO process on the specified port
        print(f"Attempting to connect TraCI on port {port}...")
        # Use a loop for connection retries as SUMO might take a moment to be ready
        retry_count = 0
        max_retries = 15  # Increased retries slightly
        while retry_count < max_retries:
            try:
                traci.init(port=port)
                print("TraCI connection successful.")
                break  # Exit retry loop on success
            except ConnectionRefusedError:
                retry_count += 1
                print(
                    f"Connection refused. Retrying in 1 second... (Attempt {retry_count}/{max_retries})"
                )
                time.sleep(1)
            except Exception as e:
                print(
                    f"An unexpected error occurred during TraCI connection attempt: {e}"
                )
                # Don't retry for unexpected errors
                if sumo_process and sumo_process.poll() is None:
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


def get_lane_index(lane_id:str):
    return lane_id.split('_')[-1]

def order_lanes_in_edge(junction_lane_map:dict[str,list[list[str]]]):
    for junction, directions in junction_lane_map.items():
        for index, direction in enumerate(directions):
            direction.sort(key=get_lane_index)
    return junction_lane_map


def calculate_neighbors(junction_id:str) -> list[int]:
    x, y = junction_id.split('_')[1], junction_id.split('_')[2]
    presence_vector = [1,1,1,1]
    if x==0:
        presence_vector[3]=0
    if x==2:
        presence_vector[2]=0
    if y==0:
        presence_vector[0]=0
    if y==2:
        presence_vector[1]=0
    return presence_vector


def get_own_state(junction_id, junction_lane_map):
    pass

def build_state_vector(junction_id, junction_lane_map):
    state = []
    padding = [0]*14
    # first own jn queues, then phase, then time spent, then neighbor map, then neighbor states (ordered)
    presence_vector = calculate_neighbors[junction_id]
    # get own queues for all lanes for junction, put them in order
    own_state = get_own_state(junction_id, junction_lane_map)
    state.extend(own_state)
    state.extend(presence_vector)
    for nbr in presence_vector:
        if nbr:
            # make nbr id first, need enumerate
            nbr_state = get_own_state(junction_id, junction_lane_map)
        else:
            nbr_state = padding
        state.extend(nbr_state)

    assert len(state)==74
    return state


    

if __name__ == "__main__":
    start_sumo()
    all_lanes = traci.lane.getIDList()
    non_internal_lanes = [lane for lane in all_lanes if not lane.startswith(":")]
    print("len non internal lanes", len(non_internal_lanes))

    tl_junctions = get_traffic_light_junction_ids_and_net_sumolib(SUMO_NET_PATH)

    junction_lane_map = build_junction_lane_mapping(tl_junctions, non_internal_lanes)
    ordered_junction_lane_map = order_lanes_in_edge(junction_lane_map)
    for junction in ordered_junction_lane_map:
        print('\n', junction)
        print(ordered_junction_lane_map[junction])
