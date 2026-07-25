import os
import shutil
import subprocess
# --- Configuration Array ---


def create_and_modify(config):
    src = config["source_dir"]
    dst = config["new_dir"]

    # 1. Copy the directory
    try:
        print(f"Copying directory '{src}' to '{dst}'...")
        shutil.copytree(src, dst)
        print("Directory copied successfully.")
    except FileExistsError:
        print(f"Error: The destination directory '{dst}' already exists. Please delete it or choose a new name.")
        return
    except FileNotFoundError:
        print(f"Error: The source directory '{src}' was not found. Please check the path.")
        return

    # 2. Process file modifications
    print("\nApplying text modifications...")
    for mod in config["modifications"]:
        # Build the exact path to the file inside the newly created directory
        target_file = os.path.join(dst, mod["filepath"])
        
        # Check if the file actually exists before trying to open it
        if not os.path.exists(target_file):
            print(f"  [!] Warning: File '{target_file}' not found. Skipping.")
            continue
        
        # Read the file's current content
        with open(target_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Apply all requested text replacements
        for old_text, new_text in mod["replacements"]:
            if old_text in content:
                content = content.replace(old_text, new_text)
                print(f"  [✓] Updated '{old_text}' to '{new_text}' in {mod['filepath']}")
            else:
                print(f"  [-] Text '{old_text}' not found in {mod['filepath']}. Skipping.")

        # Write the modified content back into the file
        with open(target_file, 'w', encoding='utf-8') as file:
            file.write(content)
            
    print("\nDone! All files successfully modified.")
    #subprocess.run(["condor_submit", "submitfile.sub"], cwd=dst, check=True)
    print("  [✓] Submission successful.\n")



if __name__ == "__main__":
    cut_types=["loose","medium","strict","stricter"]
    for i in range(len(cut_types)):
        CONFIG = {
            "source_dir": f"resnet_v6_TBHadrons24_MCElectrons_v1_2layer_samexy_cut_{cut_types[i]}",      # The existing folder you want to copy
            "new_dir": f"resnet_v6_TBHadrons24_23_MCElectrons_v1_2layer_samexy_cut_{cut_types[i]}",      # The new folder you are creating
            "modifications": [
                {
                    # First file to change
                    "filepath": "config.py",      
                    "replacements": [
                        ("PERC_TRAIN = None", "PERC_TRAIN = 42000"),
                        #("TB_USE_SAME_XY_S2=False", "TB_USE_SAME_XY_S2=True")
                        #("DEBUG = True", "DEBUG = False") # You can do multiple replacements per file
                    ]
                },
                {
                    # Second file to change (supports subdirectories inside the new folder)
                    "filepath": "mc_batch.sh", 
                    "replacements": [
                        (f'name="resnet_v6_TBHadrons24_MCElectrons_v1_2layer_samexy_cut_{cut_types[i]}"', f'name="resnet_v6_TBHadrons24_23_MCElectrons_v1_2layer_samexy_cut_{cut_types[i]}"'),
                        #("VERSION=1.0", "VERSION=1.1")
                    ]
                }
            ]
        }

        create_and_modify(CONFIG)