import matplotlib.pyplot as plt

def visualize_predictions(model, val_dataset, use_real_positions=True):
    """
    Visualize the model predictions compared to the actual labels.

    Parameters:
    - model (tf.keras.Model): The trained model.
    - val_dataset (tf.data.Dataset): The validation dataset.
    - use_real_positions (bool): Flag to indicate whether to use real positions for comparison. Default is True.
    """
    # Extract data and labels from the validation dataset
    val_data = []
    val_labels = []
    for data, labels in val_dataset:
        val_data.append(data.numpy())
        val_labels.append(labels.numpy())
    
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Make predictions using the model
    predictions = model.predict(val_data)

    # Rescale the labels and predictions if they were normalized
    max_abs_value = np.max(np.abs(val_labels))
    val_labels_rescaled = val_labels * max_abs_value
    predictions_rescaled = predictions * max_abs_value

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(val_labels_rescaled[:, 0], predictions_rescaled[:, 0], alpha=0.5, label='X coordinate')
    ax[0].plot([val_labels_rescaled[:, 0].min(), val_labels_rescaled[:, 0].max()], 
               [val_labels_rescaled[:, 0].min(), val_labels_rescaled[:, 0].max()], 'r--')
    ax[0].set_xlabel('Actual')
    ax[0].set_ylabel('Predicted')
    ax[0].set_title('X Coordinate')
    ax[0].legend()

    ax[1].scatter(val_labels_rescaled[:, 1], predictions_rescaled[:, 1], alpha=0.5, label='Y coordinate')
    ax[1].plot([val_labels_rescaled[:, 1].min(), val_labels_rescaled[:, 1].max()], 
               [val_labels_rescaled[:, 1].min(), val_labels_rescaled[:, 1].max()], 'r--')
    ax[1].set_xlabel('Actual')
    ax[1].set_ylabel('Predicted')
    ax[1].set_title('Y Coordinate')
    ax[1].legend()

    plt.suptitle('Model Predictions vs Actual Labels (Real Positions)' if use_real_positions else 'Model Predictions vs Actual Labels (ML Positions)')
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming 'model' is your trained model and 'val_dataset' is your validation dataset
# visualize_predictions(model, val_dataset, use_real_positions=True)


# Load and process the data
#file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
#train_dataset, val_dataset = load_and_process_data(file_path, use_real_positions=True)

# Assuming 'model' is your trained model
#visualize_predictions(model, val_dataset, use_real_positions=True)
