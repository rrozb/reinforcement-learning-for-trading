import os
def predict_next_sb_log_dir(base_path, prefix="PPO_"):
    # Ensure the base directory exists.
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_dirs = [d for d in os.listdir(base_path) if d.startswith(prefix)]
    existing_indices = sorted([int(d.replace(prefix, "")) for d in existing_dirs])
    next_index = existing_indices[-1] + 1 if existing_indices else 1
    return os.path.join(base_path, prefix + str(next_index))