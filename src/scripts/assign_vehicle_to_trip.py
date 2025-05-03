import xml.etree.ElementTree as ET
import random

def assign_random_vtypes_to_trips(input_filename, output_filename, vtype_ids):
    """
    Reads a SUMO routes file, assigns random vehicle types to <trip> elements,
    and saves the modified XML to a new file.

    Args:
        input_filename (str): Path to the original .rou.xml file.
        output_filename (str): Path to save the modified .rou.xml file.
        vtype_ids (list): A list of vehicle type IDs (strings) to choose from.
    """
    try:
        tree = ET.parse(input_filename)
        root = tree.getroot()
        if root.tag != 'routes':
            print(f"Warning: Root element is '{root.tag}', expected 'routes'.")

        trips = root.findall('trip')

        if not trips:
            print(f"No <trip> elements found in {input_filename}.")
            elements_to_process = [] # Keep empty if only processing trips
        else:
            print(f"Found {len(trips)} <trip> elements.")
            elements_to_process = trips


        if not vtype_ids:
            print("Error: No vehicle type IDs provided to assign.")
            return

        for element in elements_to_process:
            chosen_vtype = random.choice(vtype_ids)
            element.set('type', chosen_vtype)

        tree.write(output_filename, encoding='UTF-8', xml_declaration=True)

        print(f"Successfully assigned random vtypes to {len(elements_to_process)} trips/vehicles.")
        print(f"Modified routes saved to {output_filename}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filename}")
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    input_route_file = 'src/sumo_files/scenarios/grid_3x3_lefthand/trips_10h.rou.xml'
    output_route_file = 'src/sumo_files/scenarios/grid_3x3_lefthand/trips_with_types_10h.rou.xml'

    available_vtype_ids = ["car", "truck", "motorcycle", "bus"]

    assign_random_vtypes_to_trips(input_route_file, output_route_file, available_vtype_ids)