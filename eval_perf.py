import tensorflow as tf
from Loading_data import load_and_process_data
from plot_perf import visualize_predictions


file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
    
# Load and process the data
_, val_dataset, val_data, val_labels = load_and_process_data(file_path, use_real_positions=False)

# Load the trained model
model = tf.keras.models.load_model('final_model.keras')

# Visualize the model's performance
visualize_predictions(model, val_data, val_labels, use_real_positions=False)