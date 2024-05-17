import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def load_and_process_data(file_path, num_samples=None, energy_threshold=500, use_real_positions=True, test_size=0.2, batch_size=32):
    """
    Load data from a binary file and process it for training.

    Parameters:
    - file_path (str): Path to the binary file.
    - num_samples (int, optional): Number of samples to extract. Default is None, which means load all data.
    - energy_threshold (float): Threshold for energy to filter records. Default is 500.
    - use_real_positions (bool): Flag to choose between real positions and ML predicted positions. Default is True.
    - test_size (float): Proportion of the dataset to include in the validation split. Default is 0.2.
    - batch_size (int): Number of samples per batch. Default is 32.

    Returns:
    - train_dataset (tf.data.Dataset): Training dataset.
    - val_dataset (tf.data.Dataset): Validation dataset.
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

    # Determine the indices to sample
    if num_samples is not None and num_samples < total_records:
        indices = np.linspace(0, total_records - 1, num_samples, dtype=int)
    else:
        indices = np.arange(total_records)

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
            if use_real_positions:
                train_labels_list.append(record['positions_real'])
            else:
                train_labels_list.append(record['positions_ml'])

    # Convert lists to numpy arrays for training
    train_data = np.array(train_data_list)
    train_labels = np.array(train_labels_list)

    # Normalize train_labels to fit within 0 to 1
    train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

    # Normalize train_data to fit within 0 to 1
    train_data = train_data / np.max(train_data)

    # Standardize train_data
    scaler = StandardScaler()
    train_data_shape = train_data.shape
    train_data = train_data.reshape(-1, train_data.shape[-1])  # Flatten for scaling
    train_data = scaler.fit_transform(train_data)
    train_data = train_data.reshape(train_data_shape)

    # Outlier detection and removal
    clf = IsolationForest(contamination=0.05)  # Adjust contamination rate as needed
    outliers = clf.fit_predict(train_labels)
    mask = outliers != -1
    train_data = train_data[mask]
    train_labels = train_labels[mask]

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=test_size, random_state=42)

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Example usage
    file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
    
    # Load a specific number of samples
    train_dataset_samples, val_dataset_samples = load_and_process_data(file_path, num_samples=1000, use_real_positions=True)

    # Print some samples from the training dataset
    print("Training Data Samples:")
    for batch in train_dataset_samples.take(1):
        train_data_batch, train_labels_batch = batch
        print("Train data shape:", train_data_batch.shape)
        print("Train labels shape:", train_labels_batch.shape)
        print("Train data (first sample):", train_data_batch[0].numpy())
        print("Train labels (first sample):", train_labels_batch[0].numpy())

    # Print some samples from the validation dataset
    print("\nValidation Data Samples:")
    for batch in val_dataset_samples.take(1):
        val_data_batch, val_labels_batch = batch
        print("Validation data shape:", val_data_batch.shape)
        print("Validation labels shape:", val_labels_batch.shape)
        print("Validation data (first sample):", val_data_batch[0].numpy())
        print("Validation labels (first sample):", val_labels_batch[0].numpy())

    # Check the ranges of the data
    for batch in train_dataset_samples.take(1):
        train_data_batch, train_labels_batch = batch
        print("\nRange of training data (first batch): min =", np.min(train_data_batch.numpy()), ", max =", np.max(train_data_batch.numpy()))
        print("Range of training labels (first batch): min =", np.min(train_labels_batch.numpy()), ", max =", np.max(train_labels_batch.numpy()))

    for batch in val_dataset_samples.take(1):
        val_data_batch, val_labels_batch = batch
        print("\nRange of validation data (first batch): min =", np.min(val_data_batch.numpy()), ", max =", np.max(val_data_batch.numpy()))
        print("Range of validation labels (first batch): min =", np.min(val_labels_batch.numpy()), ", max =", np.max(val_labels_batch.numpy()))
