import tensorflow as tf
from Loading_data import load_and_process_data
from plot_perf import visualize_predictions
import numpy as np


file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
    
# Load and process the data
train_dataset, val_dataset = load_and_process_data(file_path, use_real_positions=False)

# Extract validation data and labels
val_data = np.concatenate([data.numpy() for data, _ in val_dataset], axis=0)
val_labels = np.concatenate([labels.numpy() for _, labels in val_dataset], axis=0)

# Load the trained model
model = tf.keras.models.load_model('final_model.keras')

# Visualize the model's performance
visualize_predictions(model, val_data, val_labels, use_real_positions=False)