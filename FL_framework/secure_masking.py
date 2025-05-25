import numpy as np

# Simulating Diffie-Hellman: Two seeds generate a shared secret.
# In current Secure Aggregation experiment, using XOR is one of the common simulation methods.
def get_shared_key(seed_a, seed_b):
    return seed_a ^ seed_b

# create a mask that has the same shape as the model parameters and is used to cover the model
def create_mask(shared_key, param_shapes, round_idx):
    rng = np.random.default_rng(shared_key + round_idx)
    return [rng.normal(0, 0.01, shape).astype(np.float32) for shape in param_shapes]

# Masking model parameters
def apply_mask(params, masks):
    return [p + m for p, m in zip(params, masks)]

# Subtract the corresponding total mask to get the true total model parameters and
def remove_mask(aggregated_params, masks_list):
    total_mask = [sum(mask[i] for mask in masks_list) for i in range(len(aggregated_params))]
    return [p - m for p, m in zip(aggregated_params, total_mask)]