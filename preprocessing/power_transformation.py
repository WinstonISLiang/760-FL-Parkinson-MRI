import numpy as np
import os
from tqdm import tqdm

def power_transform(data, gamma=0.5):
    """
    Apply power transformation (gamma correction) to 3D volume data
    
    Parameters:
        data (np.ndarray): Input 3D volume data
        gamma (float): Gamma value for transformation, default is 0.5
        
    Returns:
        np.ndarray: Transformed data
    """
    print("Starting power transformation...")
    # Ensure data is positive
    data = np.abs(data)
    print(f"Data shape after abs: {data.shape}")
    # Add small value to avoid log(0)
    data = data + 1e-8
    # Apply power transformation
    transformed = np.power(data, gamma)
    print("Power transformation completed")
    return transformed

def process_all_files(input_dir, output_dir, gamma=0.5):
    """
    Process all .npy files in the input directory and save transformed results
    
    Parameters:
        input_dir (str): Directory containing input .npy files
        output_dir (str): Directory to save transformed files
        gamma (float): Gamma value for power transformation
    """
    print(f"\nStarting process_all_files...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Get all .npy files
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} files to process")
    
    # Process each file
    for filename in tqdm(npy_files, desc="Processing files"):
        print(f"\nProcessing file: {filename}")
        # Load data
        input_path = os.path.join(input_dir, filename)
        print(f"Loading data from: {input_path}")
        data = np.load(input_path)
        print(f"Loaded data shape: {data.shape}")
        
        # Apply power transformation
        transformed = power_transform(data, gamma)
        
        # Save transformed data
        output_path = os.path.join(output_dir, f"power_transformed_{filename}")
        print(f"Saving transformed data to: {output_path}")
        np.save(output_path, transformed)
        print(f"Saved file: {output_path}")

if __name__ == "__main__":
    print("Script started")
    # Get current file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get project root directory (parent of current directory)
    root_dir = os.path.dirname(current_dir)
    
    # Define input and output directories
    input_dir = os.path.join(root_dir, "preprocessed_data")
    output_dir = os.path.join(root_dir, "power_transformed_data")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if os.path.exists(input_dir):
        print("Input directory exists, starting processing...")
        # Process all files
        process_all_files(input_dir, output_dir, gamma=0.5)
    else:
        print(f"Input directory not found: {input_dir}")
    print("Script completed") 