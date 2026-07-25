import torch
import os
import glob
import numpy as np

def shuffle_and_subsample_files(input_dir, file_pattern, n_samples_to_take):
    """
    Finds files, opens them individually, shuffles the data, 
    takes N samples, and saves as a new smaller file.
    """
    print(f"Scanning directory: {input_dir}")
    print(f"Looking for pattern: {file_pattern}")
    
    # 1. Find files matching the given pattern
    search_path = os.path.join(input_dir, file_pattern)
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"Error: No files found matching {search_path}")
        return

    print(f"Found {len(file_list)} files. Processing individually...\n")

    # 2. Process each file one by one
    for file_path in file_list:
        orig_name = os.path.basename(file_path)
        print(f"Loading {orig_name}...")
        
        # Load the data
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # 3. Determine total samples in this specific file
        first_val = next(iter(data.values()))
        if isinstance(first_val, torch.Tensor) or isinstance(first_val, np.ndarray):
            num_samples = first_val.shape[0]
        else:
            num_samples = len(first_val)
            
        print(f" - Found {num_samples} samples.")
        
        # Determine how many to actually take (safeguard if file has fewer than N)
        actual_n = min(n_samples_to_take, num_samples)
        
        # 4. Create shuffle indices and slice to take N
        indices = torch.randperm(num_samples)[:actual_n]
        
        subsampled_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Slicing a fixed-size tensor
                subsampled_data[key] = value[indices].clone()
            elif isinstance(value, list):
                # Slicing a variable-length python list using comprehension
                subsampled_data[key] = [value[idx.item()] for idx in indices]
            elif isinstance(value, np.ndarray):
                # Slicing a numpy array (convert indices to numpy first)
                subsampled_data[key] = value[indices.numpy()]
            else:
                # Fallback for other sequence types
                subsampled_data[key] = [value[idx.item()] for idx in indices]
        
        # 5. Save the smaller, shuffled subset
        save_name = f"shuffled_smaller_{orig_name}"
        save_path = os.path.join(input_dir, save_name)
        
        torch.save(subsampled_data, save_path)
        print(f" -> Saved {actual_n} samples to: {save_name}\n")

    print("Process completed successfully!")


# ==========================================
# Execution Configuration
# ==========================================
if __name__ == "__main__":
    OUT_DIR = "/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024"
    PATTERN = "MCEB_TB_MC_2024_electron_*.pt" 
    N_SAMPLES = 1000  # Set this to the number of samples you want to extract per file
    
    shuffle_and_subsample_files(
        input_dir=OUT_DIR, 
        file_pattern=PATTERN, 
        n_samples_to_take=N_SAMPLES
    )