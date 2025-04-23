import os
import sys
import subprocess
import time
import traci
import sumolib.net


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    if not os.path.exists(tools):
         print(f"Error: SUMO tools directory not found at '{tools}'. Please verify SUMO_HOME.")
else:
    print('SUMO_HOME environment variable not found.')
    default_sumo_home = '/usr/share/sumo'
    if os.path.exists(default_sumo_home):
        os.environ['SUMO_HOME'] = default_sumo_home
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        if tools not in sys.path:
            sys.path.append(tools)
        if not os.path.exists(tools):
            print(f"Error: SUMO tools directory not found at '{tools}' after attempting default. Please verify your SUMO installation and SUMO_HOME variable.")
    else:
        print('Could not find a default SUMO_HOME path. Please set the SUMO_HOME environment variable.')

try:
    import sumolib.net
    print("sumolib.net module imported successfully.")
except ImportError as e:
    print(f"\nError importing sumolib.net module: {e}")
    print("Please ensure SUMO_HOME is set correctly and points to a SUMO installation")
    print("that includes the Python tools.")
    sys.exit(1)


SUMO_CFG_PATH = 'src/sumo_files/scenarios/grid_3x3.sumocfg'
SUMO_NET_PATH = 'src/sumo_files/scenarios/grid_3x3.net.xml'
PORT = 8813

def get_traffic_light_junction_ids_sumolib(net_file_path):
    """
    Reads the network file using sumolib and returns IDs of traffic light junctions.
    This uses sumolib's methods, which access the static network file directly.
    """
    traffic_light_ids = []
    print(f"\nReading network file '{net_file_path}' using sumolib...")
    try:
        if not os.path.exists(net_file_path):
            print(f"Error: Network file not found at '{net_file_path}'.")
            return traffic_light_ids

        net = sumolib.net.readNet(net_file_path)
        print("Network file read successfully by sumolib.")
        all_nodes = net.getNodes()
        print(f"Found {len(all_nodes)} nodes/junctions in the .net.xml file.")

        # Filter for nodes with type 'traffic_light' using sumolib's getType()
        print("Filtering for traffic light junctions using sumolib...")
        for node in all_nodes:
            # Use sumolib's node.getType() here, NOT traci.junction.getType()
            if node.getType() == 'traffic_light':
                traffic_light_ids.append(node.getID())
                # print(f"  Identified TL junction via sumolib: {node.getID()}") # Too verbose

        print(f"Found {len(traffic_light_ids)} traffic light junctions via sumolib.")
        print(traffic_light_ids)

    except Exception as e:
        print(f"\nError reading network file with sumolib: {e}")
        print(f"Please ensure '{net_file_path}' is a valid SUMO network file and sumolib is installed/accessible.")

    return traffic_light_ids


def start_sumo(use_gui=False, sumo_seed='random', port=PORT):
    """Starts the SUMO simulation as a subprocess and connects TraCI."""
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    try:
        # Use shell=True on Windows if the binary is just a name like "sumo-gui.exe"
        # Otherwise, prefer shell=False for better security and error handling
        subprocess.run([sumo_binary, "-V"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=(sys.platform == "win32"))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nError: SUMO binary '{sumo_binary}' not found or not executable. Details: {e}")
        print("Please ensure SUMO is installed and accessible in your system's PATH.")
        if 'SUMO_HOME' in os.environ:
            print(f"Also check that '{os.path.join(os.environ.get('SUMO_HOME'), 'bin', sumo_binary)}' exists.")
        sys.exit(1)


    sumo_cmd = [sumo_binary, "-c", SUMO_CFG_PATH]
    sumo_cmd.extend(["--seed", str(sumo_seed)])
    sumo_cmd.extend(["--step-length", "0.1"])
    sumo_cmd.extend(["--time-to-teleport", "-1"])
    sumo_cmd.extend(["--remote-port", str(port)])

    print(f"\nStarting SUMO process: {' '.join(sumo_cmd)}")
    sumo_process = None

    try:
        sumo_process = subprocess.Popen(sumo_cmd)

        # Wait a moment for SUMO to start and open the port
        time.sleep(2)

        # Connect TraCI to the running SUMO process on the specified port
        print(f"Attempting to connect TraCI on port {port}...")
        # Use a loop for connection retries as SUMO might take a moment to be ready
        retry_count = 0
        max_retries = 15 # Increased retries slightly
        while retry_count < max_retries:
            try:
                traci.init(port=port)
                print("TraCI connection successful.")
                break # Exit retry loop on success
            except ConnectionRefusedError:
                retry_count += 1
                print(f"Connection refused. Retrying in 1 second... (Attempt {retry_count}/{max_retries})")
                time.sleep(1)
            except Exception as e:
                print(f"An unexpected error occurred during TraCI connection attempt: {e}")
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
            print(f"Failed to connect to TraCI on port {port} after {max_retries} attempts.")
            print("Is SUMO running and listening on the specified port?")
            print("Check SUMO's console output for errors.")
            print("------------------------")
            if sumo_process and sumo_process.poll() is None:
                sumo_process.terminate()
                sumo_process.wait()
            sys.exit(1)

        try:
            traci.simulationStep()
            print("Performed initial simulation step to confirm connection.")
            sim_time = traci.simulation.getTime()
            print(f"Current simulation time after initial step: {sim_time:.1f}s")
        except traci.exceptions.TraCIException as e:
                print(f"Error during initial simulation step after TraCI init: {e}")
                print("This might indicate an issue with the simulation file or connection loading.")
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
            raise ConnectionError("TraCI connection initialized but simulation not loaded.")

        delta_t = traci.simulation.getDeltaT()
        print(f"TraCI connected and simulation loaded. Simulation deltaT: {delta_t}s")
        return sumo_process
    except traci.exceptions.TraCIException as e:
         print("\n--- TraCI Error during init/load ---")
         print(f"A TraCI specific error occurred: {e}")
         print(f"Please ensure the SUMO config file '{SUMO_CFG_PATH}' is valid and the network can be loaded.")
         print("This error often indicates a problem with loading the network or routes after connection.")
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


def inspect_traffic_light_junctions_traci(traffic_light_junction_ids):
    """
    Uses TraCI to fetch details for a given list of traffic light junction IDs.
    Assumes the junction IDs provided are indeed traffic lights (identified by sumolib).
    Filters for incoming network edge IDs and inspects them.
    """
    print("\n--- Inspecting Traffic Light Junction Details via TraCI ---")

    print("\nNote: Traffic light junctions were identified by reading the network file using sumolib.")
    print("Now attempting to fetch dynamic details (like incoming edges, phase) using TraCI.")
    print("If you encounter 'AttributeError' here, it indicates a problem with your TraCI library installation.")
    print("------------------------------------------------------------------------------------")


    if not traffic_light_junction_ids:
        print("No traffic light junction IDs provided for inspection (none found by sumolib).")
        print("-" * 30)
        return

    print(f"Inspecting {len(traffic_light_junction_ids)} identified traffic light junctions using TraCI.")


    for junction_id in traffic_light_junction_ids:
        print(f"\n  - Inspecting Traffic Light Junction ID: '{junction_id}'")
        try:
            position = traci.junction.getPosition(junction_id)

            # Get the list which contains both edge and internal lane IDs
            mixed_incoming_ids = traci.junction.getIncomingEdges(junction_id)

            # Filter the list to keep only the actual network edge IDs
            # Network edge IDs typically do not start with a colon ':'
            incoming_edge_ids = [id for id in mixed_incoming_ids if not id.startswith(':')]
            print(f"    Position: {position}")

            if incoming_edge_ids:
                print(f"    Incoming network edges ({len(incoming_edge_ids)}):")
                for i, edge_id in enumerate(incoming_edge_ids):
                    print(f"      - Edge ID: '{edge_id}'")

            else:
                print("    No incoming network edges reported for this junction (after filtering).")
            try:
                program_id = traci.trafficlight.getProgram(junction_id)
                current_phase_index = traci.trafficlight.getPhase(junction_id)

                all_logics = traci.trafficlight.getAllProgramLogics(junction_id)

                current_state = "N/A (Logic/State not found)"
                logic_source = "N/A"

                if all_logics:
                    active_logic = None
                    for logic in all_logics:
                         if logic.programID == program_id:
                            active_logic = logic
                            logic_source = f"Program '{program_id}'"
                            break

                    if active_logic:
                        # Check if the phase index is valid for the found logic
                        if 0 <= current_phase_index < len(active_logic.phases):
                            current_state = active_logic.phases[current_phase_index].state
                            logic_source += f", Phase {current_phase_index}"
                        else:
                            current_state = f"N/A (Phase index {current_phase_index} out of bounds for program '{program_id}')"
                            logic_source += f", Invalid Phase Index {current_phase_index}"
                    else:
                        current_state = f"N/A (Could not find active logic for program '{program_id}')"
                else:
                    current_state = "N/A (No traffic light logics found for this junction)"
                    logic_source = "No logics defined"


                print("    Traffic Light Info:")
                print(f"      Current Program: '{program_id}'")
                print(f"      Current Phase Index: {current_phase_index}")
                print(f"      Current State: '{current_state}'")

            except traci.exceptions.TraCIException as e:
                print(f"    Could not fetch traffic light details for '{junction_id}' (TraCI error): {e}")
            except AttributeError as e:
                print(f"    Could not fetch traffic light details for '{junction_id}' (AttributeError): {e}")
                print("    This might indicate an issue with the TraCI library or environment.")
            except Exception as e:
                print(f"    An error occurred fetching TL info for '{junction_id}' (General error): {e}")


        except AttributeError as e:
            print(f"  AttributeError fetching general junction details for '{junction_id}': {e}")
            print("  This confirms an issue with the TraCI library or environment. Cannot get details via TraCI.")
        except traci.exceptions.TraCIException as e:
            print(f"  TraCI error fetching general junction details for '{junction_id}': {e}")
        except Exception as e:
            print(f"  General error fetching general junction details for '{junction_id}': {e}")
        print("-" * 30)


    print("\n--- Regarding Incoming Edges and Internal Junction Lanes (for Traffic Light Junctions) ---")
    if traffic_light_junction_ids:
        example_tl_junction_id = traffic_light_junction_ids[0]
        print(f"Examining junction '{example_tl_junction_id}':")
        try:
            # Get the list which contains both edge and internal lane IDs as returned by TraCI
            mixed_incoming_ids = traci.junction.getIncomingEdges(example_tl_junction_id)
            print(f"\nThe raw list of IDs returned by `traci.junction.getIncomingEdges('{example_tl_junction_id}')` is:")
            print(mixed_incoming_ids)
            print("Note: This list seems to include both network edge IDs and internal junction lane IDs.")

            # Filter the list to show only the actual network edge IDs
            example_incoming_edge_ids = [id for id in mixed_incoming_ids if not id.startswith(':')]
            print("\nFiltered list containing only incoming network EDGE IDs:")
            print(example_incoming_edge_ids)
            print("These are the standard network edges leading into the junction.")

            # Optionally, show the internal incoming lane IDs for further clarification
            # We can get these specifically using getIncomingParameter('lane')
            try:
                example_internal_lane_ids = traci.junction.getIncomingParameter(example_tl_junction_id, 'lane')
                print(f"\nThe list of internal incoming LANE IDs within the junction area (from `traci.junction.getIncomingParameter('{example_tl_junction_id}', 'lane')`) is:")
                print(example_internal_lane_ids)
                print("Note: These are internal lanes *within* the junction, used by SUMO's internal logic, not the regular network lanes.")
            except traci.exceptions.TraCIException as e:
                print(f"Could not retrieve internal incoming lane IDs for example TL junction '{example_tl_junction_id}' (TraCI error): {e}")
            except Exception as e:
                print(f"Could not retrieve internal incoming lane IDs for example TL junction '{example_tl_junction_id}' (General error): {e}")


        except traci.exceptions.TraCIException as e:
            print(f"Could not retrieve raw incoming IDs for example TL junction '{example_tl_junction_id}' (TraCI error): {e}")
        except Exception as e:
            print(f"Could not retrieve raw incoming IDs for example TL junction '{example_tl_junction_id}' (General error): {e}")

    else:
        print("Cannot demonstrate incoming edge/lane order as no traffic light junctions were identified via sumolib.")

    print("\nThe order of incoming edges and their lanes is determined by the network (.net.xml) definition")
    print("and cannot be changed via TraCI after loading. It must be defined during network generation.")

    print("\n--- Traffic Light Junction Inspection Complete ---")


def close_sumo(sumo_process=None):
    """Closes the TraCI connection and terminates the SUMO subprocess."""
    try:
        if 'traci' in sys.modules and traci.isLoaded():
            print("Closing TraCI connection...")
            traci.close()
            print("TraCI connection closed.")
        else:
            # print("TraCI connection already closed or not established.")
            pass # Be less verbose if already closed
    except Exception as e:
            print(f"Error closing TraCI: {e}")
    finally:
        if sumo_process and sumo_process.poll() is None:
            print("Terminating SUMO process...")
            sumo_process.terminate()
            try:
                sumo_process.wait(timeout=5)
                print("SUMO process terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("SUMO process did not terminate gracefully within timeout, killing it.")
                sumo_process.kill()
                sumo_process.wait()
            except Exception as e:
                print(f"Error during SUMO process termination: {e}")



if __name__=='__main__':
    traffic_light_junction_ids = get_traffic_light_junction_ids_sumolib(SUMO_NET_PATH)

    if not traffic_light_junction_ids:
        print("\nNo traffic light junctions found in the network file using sumolib. Exiting.")
        sys.exit(0) # Exit gracefully if no TLs were found

    sumo_process_handle = start_sumo(use_gui=True, sumo_seed='42', port=PORT)

    if traci.isLoaded():
        inspect_traffic_light_junctions_traci(traffic_light_junction_ids)
        print("\n--- Continuing Simulation Steps (e.g., for another 100 steps) ---")
        step = 1
        max_steps = 101
        try:
            while step < max_steps:
                if not traci.isLoaded():
                    print(f"\nTraCI connection lost at step {step}.")
                    break
                traci.simulationStep()
                step += 1
        except traci.exceptions.TraCIException as e:
            print(f"\nTraCI error during simulation step {step}: {e}")
        except Exception as e:
            print(f"\nUnexpected error during simulation step {step}: {e}")

        print(f"--- Finished Simulation Steps (reached {step} steps) ---")

        close_sumo(sumo_process_handle)
    else:
        print("\nTraCI did not connect successfully. Skipping network inspection and simulation.")