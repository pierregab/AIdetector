import numpy as np

def load_and_process_data(file_path, num_samples=1000000, energy_threshold=500):
    """
    Load data from a binary file and process it for training.

    Parameters:
    - file_path (str): Path to the binary file.
    - num_samples (int): Number of samples to extract. Default is 1000000.
    - energy_threshold (float): Threshold for energy to filter records. Default is 500.

    Returns:
    - train_data (np.ndarray): Processed training data.
    - train_labels (np.ndarray): Corresponding labels for the training data.
    """
    # Define the dtype and the structure of the record with mixed types
    dtype_here = np.dtype([
        ('charge', np.float32, (256,)),  # 256 floats for charge
        ('time', np.int32, (256,)),      # 256 integers for time, now correctly handled
        ('positions_ml', np.float32, (2,)),  # 2 floats for machine learning predicted positions
        ('positions_real', np.float32, (2,)),  # 2 floats for real positions
        ('energy', np.float32)  # 1 float for energy
    ])

    # Load the entire data from the binary file
    data = np.fromfile(file_path, dtype=dtype_here)

    # Calculate number of records in the file
    total_records = data.size

    # Calculate equidistant indices for num_samples
    indices = np.linspace(0, total_records - 1, num_samples, dtype=int)

    # Initialize lists for dynamic data collection
    train_data_list = []
    train_labels_list = []

    for idx in indices:
        record = data[idx]
        if record['energy'] <= energy_threshold:  # Check if energy is within the threshold
            charge = record['charge'].reshape(16, 16)
            time = record['time'].reshape(16, 16).astype(np.float32)  # Convert time from int to float
            # Append the reshaped data and labels to the lists
            train_data_list.append(np.stack([charge, time], axis=-1))
            train_labels_list.append(record['positions_ml'])

    # Convert lists to numpy arrays for training
    train_data = np.array(train_data_list)
    train_labels = np.array(train_labels_list)
    
    return train_data, train_labels

# Example usage
# file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
# train_data, train_labels = load_and_process_data(file_path)
