import os
import shutil
import argparse

def organize_files(base_dir):
    # Extract subfolder name from the base directory name
    base_folder_name = os.path.basename(os.path.normpath(base_dir))

    # List all files in the directory
    files = os.listdir(base_dir)

    # Iterate over each file in the directory
    for file in files:
        if file.startswith("domain_p") and file.endswith(".pddl"):
            # Extract the identifier (e.g., p230)
            identifier = file.split('_')[1].split('.')[0]
            subfolder_name = f"{base_folder_name}_{identifier}"
            subfolder_path = os.path.join(base_dir, subfolder_name)

            # Create the subfolder if it doesn't exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Move domain file to the appropriate subfolder
            shutil.move(os.path.join(base_dir, file), os.path.join(subfolder_path, "domain.pddl"))

        elif file.startswith("p") and file.endswith(".pddl"):
            # Extract the identifier (e.g., p230)
            identifier = file.split('_')[0]
            subfolder_name = f"{base_folder_name}_{identifier}"
            subfolder_path = os.path.join(base_dir, subfolder_name)

            # Create the subfolder if it doesn't exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Move problem file to the appropriate subfolder
            shutil.move(os.path.join(base_dir, file), os.path.join(subfolder_path, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize PDDL files into subfolders.")
    parser.add_argument("base_dir", type=str, help="Path to the directory containing the files to organize.")
    args = parser.parse_args()

    organize_files(args.base_dir)
    print("Files organized into subfolders successfully!")
