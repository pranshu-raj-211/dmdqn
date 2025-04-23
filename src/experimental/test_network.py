import os
import sys
import subprocess
import time
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print('SUMO_HOME environment variable not found.')
    default_sumo_home = '/usr/share/sumo'
    if os.path.exists(default_sumo_home):
        os.environ['SUMO_HOME'] = default_sumo_home
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        print(f'Attempted to set SUMO_HOME to default: {default_sumo_home}')
    else:
         print('Could not find a default SUMO_HOME path. Please set the SUMO_HOME environment variable.')
         sys.exit(1)


SUMO_CFG_PATH = 'src/sumo_files/scenarios/grid_3x3.sumocfg'

def start_sumo(use_gui=False, sumo_seed='random'):
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
    # Using 0.1 for potentially smoother simulation steps if needed later
    sumo_cmd.extend(["--step-length", "0.1"])
    sumo_cmd.extend(["--time-to-teleport", "-1"]) # Disable teleporting vehicles

    print(f"Starting SUMO: {' '.join(sumo_cmd)}")

    try:
        # Use a unique label for the TraCI connection
        # Using --remote-port 0 requires getting the assigned port
        # A simpler way with traci.start is to let it manage the process
        # If using traci.start, the --remote-port 0 might not be needed unless managing process manually
        # Let's revert to the simpler traci.start without manual port handling for now.
        # If conflicts occur, manual Popen + --remote-port + traci.init would be needed.

        # Try traci.start which manages the SUMO process lifecycle
        label = f"sim_{os.getpid()}_{int(time.time())}"
        # The command list is passed directly to traci.start
        traci.start(sumo_cmd, label=label)


        # Wait briefly for TraCI to fully connect and the simulation to load
        # isLoaded() check is more reliable
        # time.sleep(1) # Not strictly necessary with isLoaded check

        # Check if the connection was successful
        if not traci.isLoaded():
             # traci.start should raise an exception if it fails, but a belt-and-suspenders check
             raise ConnectionError("TraCI connection failed immediately after start.")

        delta_t = traci.simulation.getDeltaT()
        print(f"TraCI connected successfully with label '{label}'. Simulation deltaT: {delta_t}s")
        # When using traci.start, TraCI manages the SUMO process internally.
        # There is no external process handle to return in this case.
        return None
    except traci.exceptions.TraCIException as e:
         print("\n--- TraCI Error ---")
         print(f"A TraCI specific error occurred: {e}")
         print(f"Please ensure the SUMO config file '{SUMO_CFG_PATH}' is valid and the network can be loaded.")
         print("This error often indicates a problem with loading the network or routes.")
         print("---------------------")
         # Ensure any potential connection is closed
         if traci.isLoaded():
             traci.close()
         sys.exit(1)
    except Exception as e:
        print("\n--- General Error starting SUMO or connecting TraCI ---")
        print(e)
        print(f"Please ensure the config file exists: {SUMO_CFG_PATH}")
        print("Also check if another SUMO instance/TraCI script might be running.")
        print("----------------------------------------------------")
        # Ensure any potential connection is closed
        if traci.isLoaded():
             traci.close()
        sys.exit(1)


def inspect_network():
    """Fetches and prints details about the loaded network structure, focusing on junctions."""
    print("\n--- Inspecting Network Structure (Focus on Junctions) ---")

    junction_ids = traci.junction.getIDList()
    print(f"\nFound {len(junction_ids)} junctions.")
    if junction_ids:
        print("Junction Details:")
        for junction_id in junction_ids:
            try:
                junction_type = traci.junction.getType(junction_id)
                position = traci.junction.getPosition(junction_id)
                # traci.junction.getIncomingParameter(junctionID, "lane") returns a list of incoming lane IDs
                incoming_lanes = traci.junction.getIncomingParameter(junction_id, "lane")

                print(f"\n  - Junction ID: '{junction_id}'")
                print(f"    Type: '{junction_type}'")
                print(f"    Position: {position}")

                if incoming_lanes:
                    print(f"    Incoming Lanes ({len(incoming_lanes)}):")
                    # Print each incoming lane ID and potentially some details
                    for i, lane_id in enumerate(incoming_lanes):
                        try:
                            lane_length = traci.lane.getLength(lane_id)
                            lane_speed = traci.lane.getMaxSpeed(lane_id)
                            # You could add lane shape, parent edge etc. if needed
                            print(f"      {i}: Lane ID: '{lane_id}', Length: {lane_length:.2f}m, Speed: {lane_speed:.2f}m/s")
                        except Exception as e:
                            print(f"      {i}: Could not fetch details for incoming lane '{lane_id}': {e}")
                else:
                    print("    No incoming lanes reported for this junction.")

                if junction_type == 'traffic_light':
                    try:
                        current_phase_index = traci.trafficlight.getPhase(junction_id)
                        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)
                        current_state = "N/A (Logic not found)"
                        program_id = traci.trafficlight.getProgram(junction_id)

                        if logic:
                             for phase in logic[0].phases:
                                 if phase.phase == current_phase_index: # Phase is stored by index in logic
                                     current_state = phase.state
                                     break # Found the state for the current phase

                        print("    Traffic Light Info:")
                        print(f"      Current Program: '{program_id}'")
                        print(f"      Current Phase Index: {current_phase_index}")
                        print(f"      Current State: '{current_state}'")

                    except traci.exceptions.TraCIException as e:
                         # This might happen if the junction is marked traffic_light but has no logic loaded
                         print(f"    Could not fetch traffic light details for '{junction_id}': {e}")
                    except Exception as e:
                         print(f"    An error occurred fetching TL info for '{junction_id}': {e}")

                # Add a separator for readability between junctions
                print("-" * 30)


            except Exception as e:
                print(f"  - Could not fetch general details for junction '{junction_id}': {e}")

    else:
        print("No junctions found in the network.")

    # --- Addressing the Incoming Lane Order Question ---
    print("\n--- Regarding Incoming Lane Order ---")
    print("The list of incoming lane IDs returned by `traci.junction.getIncomingParameter(junctionID, 'lane')`")
    print(f"for a junction like '{junction_ids[0]}' (example) is: {traci.junction.getIncomingParameter(junction_ids[0], 'lane') if junction_ids else 'N/A'}")
    print("\nThis order is determined by the SUMO network (.net.xml) file definition.")
    print("It typically corresponds to the order in which the connections from these lanes")
    print("to the junction were defined during the network building process (e.g., by netconvert).")
    print("This order is **consistent** for a given network file.")
    print("\n**Can an order be imposed?**")
    print("Via TraCI commands *after* the network is loaded: **No**, you cannot reorder the lanes")
    print("in the list returned by `getIncomingParameter` or fundamentally change their internal representation order.")
    print("During network generation (e.g., using netconvert, or editing SUMO network files like .net.xml): **Yes**, the order in which")
    print("connections or edges/lanes are defined in the input files (.net.xml, .osm.xml converted via netconvert, etc.)")
    print("influences the resulting order in the loaded network and thus the order seen via TraCI.")
    print("If you need a specific, predictable order (e.g., clockwise or counter-clockwise), you must ensure")
    print("your network generation process produces the `.net.xml` file with connections/lanes defined in that desired order.")

    print("\n--- Network Inspection Complete ---")


def close_sumo(sumo_process=None):
    """Closes the TraCI connection."""
    # When using traci.start, TraCI manages the SUMO process.
    # Calling traci.close() is usually sufficient.
    try:
        if traci.isLoaded(): # Use isLoaded() to check connection status
            print("Closing TraCI connection...")
            traci.close()
            print("TraCI connection closed.")
        else:
            print("TraCI connection already closed or not established.")
    except NameError: # Handle case where traci wasn't imported successfully
            print("TraCI module not available.")
    except Exception as e:
            print(f"Error closing TraCI: {e}")
    # If traci.start was used, there's no external process to terminate here


if __name__=='__main__':
     sumo_process_handle = start_sumo(use_gui=True, sumo_seed='42')

     # Check if TraCI is connected before proceeding
     if traci.isLoaded():
         # Run network inspection
         inspect_network()

         # You could potentially run simulation steps here using traci.simulationStep()
         # For this script, we just inspect and close.
         # print("\nRunning a few simulation steps (e.g., 10 steps)...")
         # for step in range(10):
         #    traci.simulationStep()
         #    # print(f"Simulation step {step}") # Too verbose, maybe print less often
         # print("Simulation steps finished.")


         # Close SUMO simulation via TraCI
         # traci.close() is the standard way to stop the simulation started by traci.start
         close_sumo(sumo_process_handle)
     else:
         print("\nTraCI did not connect. Skipping network inspection and close.")