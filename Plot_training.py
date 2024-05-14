import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plots the training and validation loss and mean absolute error (MAE) from the training history.

    Parameters:
    - history: A Keras History object. Contains training and validation loss and MAE values for each epoch.

    Returns:
    - None
    """
    # Extracting the metrics history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)

    # Plotting Loss
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Mean Absolute Error
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae, 'b-', label='Training MAE')
    plt.plot(epochs, val_mae, 'r-', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
# Assuming you have a trained model and its history object
# plot_training_history(history)
