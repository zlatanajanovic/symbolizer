import sys
import json
import os
import yaml
import random
from typing import Dict, Any
from pddl import parse_domain

def load_dataset(file_path: str) -> Dict[str, Any]:
    """Loads the dataset from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


# Load config.yaml
config_path = "config_eval.yaml"  # Adjust if needed
if not os.path.isfile(config_path):
    print(f"Error: Config file '{config_path}' not found.")
    sys.exit(1)

with open(config_path, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        sys.exit(1)


object_response_file = config["objects_full_responses_path"]
output_pddl_folder = os.path.join(str(config["output_folder"]), "problems/")

dataset_path = config["dataset_path"]
if not os.path.isfile(dataset_path):
    print(f"Error: Dataset file '{dataset_path}' not found.")
    sys.exit(1)

dataset_0 = load_dataset(dataset_path)
dataset = dataset_0.get("problems", [])

# Set the fixed number of duplicates
num_duplicate = config.get("num_answers_per_sample", 1)

# Load all request_ids from the object_response_file
request_ids = []
with open(object_response_file, 'r') as infile:
    for line_num, line in enumerate(infile, 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            line_request_id = data.get('request_id')
            if line_request_id:
                request_ids.append(line_request_id)
            else:
                print(f"Warning: Line {line_num} in {object_response_file} missing 'request_id'. Skipped.")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON on line {line_num} in {object_response_file}: {e}. Skipped.")
request_ids = list(set(request_ids))
if not request_ids:
    print("Error: No valid 'request_id's found in the object response file.")
    sys.exit(1)

# Extract necessary paths from config
objects_full_responses_path = config.get('objects_full_responses_path')
atoms_full_responses_path = config.get('atoms_full_responses_path')
goals_full_responses_path = config.get('goals_full_responses_path')

dict_request_id_to_problem_order = {}
for request_id_unique in request_ids:
    try:
        problem_position = int(request_id_unique.split("problem_")[1].split("_state")[0])
    except (IndexError, ValueError) as e:
        print(f"Warning: Unable to extract problem position from request_id '{request_id_unique}': {e}. Skipped.")
        continue

    if problem_position >= len(dataset):
        print(f"Warning: problem_position {problem_position} out of range for dataset.")
        continue

    problem_entry = dataset[problem_position]
    try:
        domain_name_request_id = request_id_unique.split("_domain_")[1]
    except IndexError:
        print(f"Warning: Unable to extract domain name from request_id '{request_id_unique}'. Skipped.")
        continue

    domain_name_dataset = problem_entry.get("domain_name")
    if domain_name_request_id != domain_name_dataset:
        raise ValueError(f"Domain name mismatch between request_id '{domain_name_request_id}' and dataset '{domain_name_dataset}'")

    problem_file_path = problem_entry.get("problem_file", "")
    if not problem_file_path:
        print(f"Warning: 'problem_file' missing in dataset entry for problem_position {problem_position}. Skipped.")
        continue

    problem_name_file_name = os.path.splitext(os.path.basename(problem_file_path))[0]

    existing_entry = dict_request_id_to_problem_order.get(request_id_unique)
    if existing_entry:
        if existing_entry["problem_file_name"] != problem_name_file_name:
            raise ValueError(f"Problem file name mismatch between request_id '{request_id_unique}' and dataset '{problem_name_file_name}'")
    else:
        dict_request_id_to_problem_order[request_id_unique] = {
            "domain_name": domain_name_dataset,
            "problem_file_name": problem_name_file_name
        }

if not all([objects_full_responses_path, atoms_full_responses_path, goals_full_responses_path]):
    print("Error: One or more required paths ('objects_full_responses_path', 'atoms_full_responses_path', 'goals_full_responses_path') are missing in config.yaml.")
    sys.exit(1)


# Helper function to load entry by request_id
def load_entry_by_request_id(jsonl_file, key="request_id", desired_id=None):
    if not os.path.isfile(jsonl_file):
        print(f"Error: File not found: {jsonl_file}")
        return None
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num} in {jsonl_file}: {e}. Skipped.")
                continue
            if data.get(key) == desired_id:
                return data
    return None


# Dictionary to keep track of how many duplicates have been created for each problem_file_name
problem_file_name_counts = {}

# Keep track of domain directories we have processed
processed_domains = set()

# Iterate over the requests
for request_id in request_ids:
    # Extract domain name from request_id
    if '_domain_' not in request_id:
        print(f"Error: request_id '{request_id}' does not contain '_domain_'. Skipped.")
        continue  # Skip this request_id

    try:
        domain_name = request_id.split('_domain_')[1]
    except IndexError:
        print(f"Error: Unable to extract domain name from request_id '{request_id}'. Skipped.")
        continue

    # Load objects, atoms, and goals entries
    objects_data = load_entry_by_request_id(objects_full_responses_path, "request_id", request_id)
    atoms_data = load_entry_by_request_id(atoms_full_responses_path, "request_id", request_id)
    goals_data = load_entry_by_request_id(goals_full_responses_path, "request_id", request_id)

    if objects_data is None:
        print(f"Error: Could not find entries for request_id '{request_id}' in the Object JSONL files.")
        continue  # Skip this request_id
    if atoms_data is None:
        print(f"Error: Could not find entries for request_id '{request_id}' in the Atoms JSONL files.")
        continue
    if goals_data is None:
        # Get the ground truth goals from the problem_file
        try:
            problem_position = int(request_id.split("problem_")[1].split("_state")[0])
            problem_entry = dataset[problem_position]
            goal_data = json.loads(problem_entry["states"][0].get("goal_predicates", "{}"))
            goals_data = {
                "request_id": request_id,
                "goal_predicates": goal_data.get("grounded_predicates", []),
                "domain_file": problem_entry.get("domain_file", "")
            }
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"Error processing goals for request_id '{request_id}': {e}. Skipped.")
            continue
        
    # we are not using the vlm to parse goals at any point Thus we have to simulate the enforcing of  structure  
    object_names = [o["name"] for o in objects_data["objects"]] 
    new_goals_predicates = []
    for predicate_element in goals_data["goal_predicates"]:
        predicate_dict = {}
        x_name = None
        y_name = None
        z_name = None   
        predicate_dict["predicate_type"] = predicate_element["predicate_type"]
        
        if "x" in predicate_element:
            try:
                x_name = predicate_element["x"]["name"]
            except KeyError:
                continue
            if x_name not in object_names:
                continue
            else:
                predicate_dict["x"] = {"name": x_name}
        if "y" in predicate_element:
            y_name = predicate_element["y"]["name"]
            if y_name not in object_names:
                continue
            else:
                predicate_dict["y"] = {"name": y_name}
        if "z" in predicate_element:
            z_name = predicate_element["z"]["name"]
            if z_name not in object_names:
                continue
            else:
                predicate_dict["z"] = {"name": z_name}
        new_goals_predicates.append(predicate_dict)
    if len(new_goals_predicates) == 0:
        print(f"Warning: No valid goal predicates found for request_id '{request_id}'. Will be markes as syntax error.")
    else:
        goals_data["goal_predicates"] = new_goals_predicates


    domain_file = objects_data.get("domain_file")
    if not domain_file:
        print(f"Error: No 'domain_file' found in objects_data for request_id '{request_id}'. Skipped.")
        continue  # Skip this request_id

    # Check if domain_file exists
    if not os.path.isfile(domain_file):
        print(f"Error: domain_file '{domain_file}' does not exist for request_id '{request_id}'. Skipped.")
        continue  # Skip this request_id

    # Parse the domain
    try:
        domain = parse_domain(domain_file)
    except Exception as e:
        print(f"Error parsing domain file '{domain_file}' for request_id '{request_id}': {e}")
        continue  # Skip this request_id

    # Extract objects, atoms, and goals
    objects = objects_data.get("objects", [])
    atoms = atoms_data.get("grounded_predicates", [])
    goals = goals_data.get("goal_predicates", [])

    # Convert objects to PDDL objects section
    type_map = {}
    for obj in objects:
        o_name = obj.get('name')
        o_type = obj.get('type')
        if not o_name or not o_type:
            print(f"Warning: Object entry missing 'name' or 'type' in request_id '{request_id}'. Skipped object.")
            continue
        type_map.setdefault(o_type, []).append(o_name)

    objects_str = ""
    for t, o_list in type_map.items():
        objects_str += " ".join(o_list) + " - " + t + "\n    "

    # Convert atoms to PDDL init
    init_str = ""
    for atom in atoms:
        pred = atom.get("predicate_type")
        if not pred:
            print(f"Warning: Atom entry missing 'predicate_type' in request_id '{request_id}'. Skipped atom.")
            continue
        arg_names = []
        # Gather args x, y, z... in alphabetical order
        for k in sorted(atom.keys()):
            if k == "predicate_type":
                continue
            val = atom[k]
            if isinstance(val, dict) and "name" in val:
                arg_names.append(val["name"])
        if arg_names:
            init_str += f"({pred} {' '.join(arg_names)})\n    "
        else:
            print("atom with no args: ", atom)
            #add it as it dies not require any args
            init_str += f"({pred})\n    "

    # Convert goals to PDDL goal
    if not goals:
        goal_str = ""
    elif len(goals) == 1:
        g = goals[0]
        pred = g.get("predicate_type")
        if not pred:
            print(f"Warning: Goal entry missing 'predicate_type' in request_id '{request_id}'. Skipped goal.")
            goal_str = ""
        else:
            arg_names = []
            for k in sorted(g.keys()):
                if k == "predicate_type":
                    continue
                val = g[k]
                if isinstance(val, dict) and "name" in val:
                    arg_names.append(val["name"])
            if arg_names:
                goal_str = f"({pred} {' '.join(arg_names)})"
            else:
                print("goal with no args: ", g)
                #add it as it dies not require any args
    else:
        goal_str = "(and "
        for g in goals:
            pred = g.get("predicate_type")
            if not pred:
                print(f"Warning: Goal entry missing 'predicate_type' in request_id '{request_id}'. Skipped goal.")
                continue
            arg_names = []
            for k in sorted(g.keys()):
                if k == "predicate_type":
                    continue
                val = g[k]
                if isinstance(val, dict) and "name" in val:
                    arg_names.append(val["name"])
            if arg_names:
                goal_str += f"({pred} {' '.join(arg_names)}) "
        goal_str += ")"

    # Create domain subfolder if it doesn't exist
    domain_path = os.path.join(output_pddl_folder, domain_name)
    os.makedirs(domain_path, exist_ok=True)

    # Create problems subfolder if it doesn't exist
    problems_path = os.path.join(domain_path, "problems")
    os.makedirs(problems_path, exist_ok=True)
    processed_domains.add(domain_path)

    # Fetch the problem_file_name from dict_request_id_to_problem_order
    problem_info = dict_request_id_to_problem_order.get(request_id)
    if not problem_info:
        print(f"Error: No problem information found for request_id '{request_id}'. Skipped.")
        continue

    problem_file_name = problem_info["problem_file_name"]

    # Initialize count if not present
    if problem_file_name not in problem_file_name_counts:
        problem_file_name_counts[problem_file_name] = 0

    # Generate num_duplicate PDDL problem files, each with incremented numbering
    for _ in range(num_duplicate):
        # Increment count for this problem_file_name
        problem_file_name_counts[problem_file_name] += 1

        pddl_content = (
            f"(define (problem {problem_file_name})\n"
            f"  (:domain {domain.name})\n"
            f"  (:objects\n    {objects_str})\n"
            f"  (:init\n    {init_str})\n"
            f"  (:goal\n    {goal_str}))"
        )

        # Make the PDDL content lowercase
        pddl_content = pddl_content.lower()

        # For now, just name them uniquely with a large random number to avoid any collisions.
        rand_suffix = random.randint(100000000, 999999999)
        output_pddl_file = os.path.join(problems_path, f"{problem_file_name}_{rand_suffix}.pddl")
        try:
            with open(output_pddl_file, 'w') as f_out:
                f_out.write(pddl_content)
            print(f"Successfully wrote {output_pddl_file}")
        except Exception as e:
            print(f"Error writing to file '{output_pddl_file}': {e}")
            continue  # Proceed to the next problem

print("Generation of PDDL problem files completed.")

########################################
# Post-processing step:
# 1. For each processed domain_path, go into its problems directory.
# 2. Extract the problem name from (define (problem XXX)) line in each .pddl file.
# 3. Sort files by that problem name.
# 4. Rename files to problem.1.pddl, problem.2.pddl, etc.
########################################

for dpath in processed_domains:
    problems_path = os.path.join(dpath, "problems")
    if not os.path.isdir(problems_path):
        continue

    # Gather all pddl files
    pddl_files = [f for f in os.listdir(problems_path) if f.endswith(".pddl")]
    file_info = []
    for pfile in pddl_files:
        full_path = os.path.join(problems_path, pfile)
        # Extract problem name from inside the file
        # We're looking for a line like: (define (problem something))
        problem_name = None
        with open(full_path, 'r') as f_in:
            for line in f_in:
                line = line.strip().lower()
                if line.startswith("(define (problem"):
                    # Extract the problem name
                    # line might look like: (define (problem problem123)
                    # Let's split by spaces:
                    parts = line.split()
                    # parts[2] should be (problem_name)
                    # structure: (define (problem XXX)
                    # parts[2] = (problem
                    # parts[3] = XXX)
                    # safer approach: find substring between (problem and )
                    if "(problem" in line:
                        # Extract the substring after (problem
                        after_problem = line.split("(problem", 1)[1].strip()
                        # after_problem might be like " problem123)"
                        after_problem = after_problem.strip(" )")
                        problem_name = after_problem
                    break

        if problem_name is None:
            print(f"Warning: Could not extract a problem name from {full_path}, skipping.")
            continue

        file_info.append((problem_name, full_path))

    # Sort files by the extracted problem name, but remove all non-number characters
    file_info.sort(key=lambda x: int("".join(filter(str.isdigit, x[0]))))
    # Now rename them in order: problem.1.pddl, problem.2.pddl, ...
    # We'll just enumerate them.
    for idx, (_, old_path) in enumerate(file_info, start=1):
        new_name = f"problem{idx}.pddl"
        new_path = os.path.join(problems_path, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

print("Final renaming and sorting of PDDL files completed.")
