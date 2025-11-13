import time
import os
import scipy.io
import numpy as np

# Set random seed for reproducibility
np.random.seed(1000)

# Load the original dense tensor
dense_tensor = scipy.io.loadmat('./Guangzhou-data-set/tensor.mat')['tensor']

# Get tensor dimensions
dim = dense_tensor.shape

# Missing rate
missing_rate = 0.4  # Random missing (RM)

# Generate sparse tensor with missing values
sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate)

# Print the tensor dimensions
print(dim)

# Define the output directory
save_dir = './Guangzhou-data-setMitMissingValue'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it does not exist

# Save the sparse tensor as a .mat file
scipy.io.savemat(os.path.join(save_dir, 'sparse_tensor.mat'), {'tensor': sparse_tensor})
#只需要在sample_factor那一步不更新factor就可以固定factor了
