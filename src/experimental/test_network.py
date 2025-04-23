import os
import sys
import subprocess
import time
import traci
import sumolib.net # Import sumolib.net

# Ensure SUMO_HOME is set and add tools to sys.path
# (SUMO_HOME and sys.path logic remains the same as previous responses)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
        #print(f"Added '{tools}' to sys.path") # Keep prints less verbose unless debugging
    # else:
        #print(f"'{tools}' already in sys.path")

    if not os.path.exists(tools):
         print(f"Error: SUMO tools directory not found at '{tools}'. Please verify SUMO_HOME.")
         # sys.exit(1) # Don't exit immediately, allow printing sys.path
else:
    print('SUMO_HOME environment variable not found.')
    default_sumo_home = '/usr/share/sumo'
    if os.path.exists(default_sumo_home):
        os.environ['SUMO_HOME'] = default_sumo_home
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        if tools not in sys.path:
            sys.path.append(tools)
            #print(f'Attempted to set SUMO_HOME to default: {default_sumo_home}')
            #print(f"Added '{tools}' to sys.path")
        # else:
             #print(f"Attempted to set SUMO_HOME to default: {default_sumo_home}, '{tools}' already in sys.path")

        if not os.path.exists(tools):
             print(f"Error: SUMO tools directory not found at '{tools}' after attempting default. Please verify your SUMO installation and SUMO_HOME variable.")
             # sys.exit(1) # Don't exit immediately, allow printing sys.path
    else:
         print('Could not find a default SUMO_HOME path. Please set the SUMO_HOME environment variable.')
         # sys.exit(1) # Don't exit immediately, allow printing sys.path

# Print sys.path to help diagnose import issues (optional, uncomment for debugging)
# print("\nCurrent sys.path:")
# for path in sys.path:
#     print(f"  {path}")
# print("-" * 20)

# Attempt to import traci and sumolib after setting sys.path
try:
    import traci
    print("traci module imported successfully.")
    # Optional: Try a very basic call that might reveal early issues
    # try:
    #     traci_version = traci.getVersion()
    #     print(f"TraCI library version reported: {traci_version}")
    # except Exception as e:
    #     print(f"Could not retrieve TraCI library version: {e}")

except ImportError as e:
    print(f"\nError importing traci module: {e}")
    print("Please ensure SUMO_HOME is set correctly and points to a SUMO installation")
    print("that includes the Python tools.")
    sys.exit(1)

try:
    import sumolib.net
    print("sumolib.net module imported successfully.")
except ImportError as e:
     print(f"\nError importing sumolib.net module: {e}")
     print("Please ensure SUMO_HOME is set correctly and points to a SUMO installation")
     print("that includes the Python tools.")
     sys.exit(1)


# --- Configuration ---
# Make sure this path is correct relative to where you run the script
SUMO_CFG_PATH = 'src/sumo_files/scenarios/grid_3x3.sumocfg'
# We need the .net.xml path to read with sumolib
# Assuming the .net.xml is in the same directory as the .sumocfg and has a similar name
# You might need to adjust this path based on your actual file structure
SUMO_NET_PATH = 'src/sumo_files/scenarios/grid_3x3.net.xml' # *** Adjust if necessary ***

# Define a fixed port for TraCI connection when using Popen
PORT = 8813
# ---------------------

def get_traffic_light_junction_ids_sumolib(net_file_path):
    """
    Reads the network file using sumolib and returns IDs of traffic light junctions.
    This uses sumolib's methods, which access the static network file directly.
    """
    traffic_light_ids = []
    print(f"\nReading network file '{net_file_path}' using sumolib...")
    try:
        # Check if the network file exists
        if not os.path.exists(net_file_path):
            print(f"Error: Network file not found at '{net_file_path}'.")
            return traffic_light_ids # Return empty list

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
    # Check if the binary exists and is executable
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


    # Build the SUMO command to be run as a subprocess
    sumo_cmd = [sumo_binary, "-c", SUMO_CFG_PATH]
    sumo_cmd.extend(["--seed", str(sumo_seed)])
    sumo_cmd.extend(["--step-length", "0.1"])
    sumo_cmd.extend(["--time-to-teleport", "-1"]) # Disable teleporting vehicles
    # Specify the remote port for TraCI connection
    sumo_cmd.extend(["--remote-port", str(port)])

    print(f"\nStarting SUMO process: {' '.join(sumo_cmd)}")
    sumo_process = None

    try:
        # Start SUMO as a separate process
        sumo_process = subprocess.Popen(sumo_cmd)

        # Wait a moment for SUMO to start and open the port
        time.sleep(2) # Increased wait time slightly

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
             print(f"\n--- Connection Error ---")
             print(f"Failed to connect to TraCI on port {port} after {max_retries} attempts.")
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
             print("This might indicate an issue with the simulation file or connection loading.")
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
             # This check should ideally not be needed after a successful init and step
             raise ConnectionError("TraCI connection initialized but simulation not loaded.")

        delta_t = traci.simulation.getDeltaT()
        print(f"TraCI connected and simulation loaded. Simulation deltaT: {delta_t}s")
        return sumo_process
    except traci.exceptions.TraCIException as e:
         print(f"\n--- TraCI Error during init/load ---")
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
    """
    print("\n--- Inspecting Traffic Light Junction Details via TraCI ---")

    print("\nNote: Traffic light junctions were identified by reading the network file using sumolib.")
    print("Now attempting to fetch dynamic details (like incoming lanes, phase) using TraCI.")
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
            # Attempt to get details using TraCI
            # These calls might still fail with AttributeError if the TraCI domain objects are broken
            position = traci.junction.getPosition(junction_id)
            incoming_edges = traci.junction.getIncomingEdges(junction_id)
            print(incoming_edges)
            print(f"    Position: {position}")
            if incoming_edges:
                print(f"    Incoming edges ({len(incoming_edges)}):")
                for i, edge_id in enumerate(incoming_edges):
                    try:
                        # Attempt to get edge details - these calls might also fail
                        #lane_length = traci.edge.getLength(edge_id[1:])
                        #lane_speed = traci.edge.getMaxSpeed(edge_id[1:])
                        #print(f"      {i}: Lane ID: '{edge_id}', Speed: {lane_speed:.2f}m/s")
                        pass
                    except AttributeError as e:
                        print(f"      {i}: Could not fetch details for incoming edge '{edge_id}' (AttributeError): {e}")
                        print("      TraCI library issue affects edge details too.")
                    except traci.exceptions.TraCIException as e:
                        print(f"      {i}: Could not fetch details for incoming edge '{edge_id}' (TraCI error): {e}")
                    except Exception as e:
                        print(f"      {i}: Could not fetch details for incoming edge '{edge_id}' (General error): {e}")
            else:
                print("    No incoming edge reported for this junction.")

            
            try:
                # These calls use the trafficlight domain, which might work or also fail
                # with AttributeError if the problem is systemic in TraCI
                current_phase_index = traci.trafficlight.getPhase(junction_id)
                logic = traci.trafficlight.getAllProgramLogics(junction_id)
                current_state = "N/A (Logic/State not found)"
                program_id = traci.trafficlight.getProgram(junction_id)

                if logic and len(logic) > 0:
                    if current_phase_index < len(logic[0].phases):
                        current_state = logic[0].phases[current_phase_index].state
                    else:
                        current_state = f"N/A (Phase index {current_phase_index} out of bounds for current logic)"
                else:
                    current_state = "N/A (No traffic light logic found for this junction)"


                print(f"    Traffic Light Info:")
                print(f"      Current Program: '{program_id}'")
                print(f"      Current Phase Index: {current_phase_index}")
                print(f"      Current State: '{current_state}'")

            except AttributeError as e:
                print(f"    Could not fetch traffic light details for '{junction_id}' (AttributeError): {e}")
                print("    This indicates an issue with the TraCI library or environment.")
            except traci.exceptions.TraCIException as e:
                print(f"    Could not fetch traffic light details for '{junction_id}' (TraCI error): {e}")
            except Exception as e:
                print(f"    An error occurred fetching TL info for '{junction_id}' (General error): {e}")


        except AttributeError as e:
            print(f"  AttributeError fetching details for junction '{junction_id}': {e}")
            print("  This confirms an issue with the TraCI library or environment. Cannot get details via TraCI.")
        except traci.exceptions.TraCIException as e:
            print(f"  TraCI error fetching details for junction '{junction_id}': {e}")
        except Exception as e:
            print(f"  General error fetching details for junction '{junction_id}': {e}")

        print("-" * 30)


    # --- Addressing the Incoming Lane Order Question (for TLs if found) ---
    print("\n--- Regarding Incoming Lane Order (for Traffic Light Junctions) ---")
    if traffic_light_junction_ids:
        # Pick the first traffic light junction from the list
        example_tl_junction_id = traffic_light_junction_ids[0]
        try:
            # Attempt to get incoming lanes for the example TL junction using TraCI
            # This call might still fail with AttributeError
            example_lanes = traci.junction.getIncomingEdges(example_tl_junction_id)
            print(f"The list of incoming lane IDs returned by `traci.junction.getIncomingParameter(junctionID, 'lane')`")
            print(f"for an example traffic light junction like '{example_tl_junction_id}' is: {example_lanes}")
        except AttributeError as e:
            print(f"Could not retrieve incoming lanes for example TL junction '{example_tl_junction_id}' (AttributeError): {e}")
            print("This confirms the TraCI library issue prevents showing the example order via TraCI.")
        except traci.exceptions.TraCIException as e:
            print(f"Could not retrieve incoming lanes for example TL junction '{example_tl_junction_id}' (TraCI error): {e}")
        except Exception as e:
            print(f"Could not retrieve incoming lanes for example TL junction '{example_tl_junction_id}' (General error): {e}")

    else:
        print("Cannot demonstrate incoming lane order as no traffic light junctions were identified via sumolib.")


    print("\nAs mentioned for general junctions, the order is determined by the network (.net.xml) definition")
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


# --- Main Execution ---
if __name__=='__main__':
     # --- Use sumolib to identify traffic light junctions first ---
     # Make sure SUMO_NET_PATH is correct!
    traffic_light_junction_ids = get_traffic_light_junction_ids_sumolib(SUMO_NET_PATH)

    if not traffic_light_junction_ids:
        print("\nNo traffic light junctions found in the network file using sumolib. Exiting.")
        sys.exit(0) # Exit gracefully if no TLs were found

     # --- Start SUMO and connect TraCI ---
    sumo_process_handle = start_sumo(use_gui=True, sumo_seed='42', port=PORT)

     # Check if TraCI is connected before proceeding with TraCI calls
    if traci.isLoaded():
         # --- Inspect identified traffic light junctions using TraCI ---
         # Pass the list of TL IDs found by sumolib
        inspect_traffic_light_junctions_traci(traffic_light_junction_ids)

         # --- Simulation Loop Example ---
         # (Simulation loop remains the same as the previous response)
        print("\n--- Continuing Simulation Steps (e.g., for another 100 steps) ---")
        step = 1 # We already did one step in start_sumo
        max_steps = 101 # Run until step 100 (total 100 steps after the initial one)
        try:
            while step < max_steps:
                if not traci.isLoaded():
                    print(f"\nTraCI connection lost at step {step}.")
                    break
                traci.simulationStep()
                # sim_time = traci.simulation.getTime()
                # if step % 10 == 0:
                #     print(f"Simulation time: {sim_time:.1f}s")

                step += 1
        except traci.exceptions.TraCIException as e:
            print(f"\nTraCI error during simulation step {step}: {e}")
        except Exception as e:
            print(f"\nUnexpected error during simulation step {step}: {e}")

        print(f"--- Finished Simulation Steps (reached {step} steps) ---")


        # Close SUMO simulation and terminate the process
        close_sumo(sumo_process_handle)
    else:
        print("\nTraCI did not connect successfully. Skipping network inspection and simulation.")