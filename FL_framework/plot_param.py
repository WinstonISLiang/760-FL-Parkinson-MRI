import numpy as np
import matplotlib.pyplot as plt
import os

# see the result of client1
tmp_dir = "tmp_param"
client_id = 1

param = np.load(os.path.join(tmp_dir, f"param_client{client_id}.npy")).flatten()
masked_param = np.load(os.path.join(tmp_dir, f"masked_param_client{client_id}.npy")).flatten()

plt.figure(figsize=(8, 5))
plt.hist(param, bins=50, alpha=0.7, label='Parameter')
plt.hist(masked_param, bins=50, alpha=0.7, label='Masked Parameter')
plt.xlabel('Parameter Value')
plt.ylabel('Frequency')
plt.title(f'Client {client_id} Parameter Distribution (original vs Masked)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(tmp_dir, f'param_distribution_compare_client{client_id}.png'), dpi=150)
plt.show()
