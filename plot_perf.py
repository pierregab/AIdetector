import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def visualize_predictions(model, val_data, val_labels, use_real_positions=True, num_samples=10000):
    """
    Visualize the model predictions compared to the actual labels with random sample selection.

    Parameters:
    - model (tf.keras.Model): The trained model.
    - val_data (np.ndarray): The validation data.
    - val_labels (np.ndarray): The validation labels.
    - use_real_positions (bool): Flag to indicate whether to use real positions for comparison. Default is True.
    - num_samples (int): Number of random samples to select for visualization. Default is 10000.
    """
    # Select a random sample or batch from the dataset
    sample_indices = np.random.choice(val_data.shape[0], size=num_samples, replace=False)
    sample_data = val_data[sample_indices]
    sample_labels = val_labels[sample_indices]

    # Make predictions using the model
    predicted_labels = model.predict(sample_data)

    # Rescale the labels and predictions if they were normalized
    max_abs_value = np.max(np.abs(val_labels))
    sample_labels_rescaled = sample_labels * max_abs_value
    predicted_labels_rescaled = predicted_labels * max_abs_value

    # Calculate percentage errors
    percentage_errors = 100 * (predicted_labels_rescaled - sample_labels_rescaled) / sample_labels_rescaled  # element-wise operation

    # Calculate absolute errors
    absolute_errors = np.abs(predicted_labels_rescaled - sample_labels_rescaled)

    # Extract x and y percentage errors
    x_errors = percentage_errors[:, 0]
    y_errors = percentage_errors[:, 1]

    # Extract x and y absolute errors
    x_abs_errors = absolute_errors[:, 0]
    y_abs_errors = absolute_errors[:, 1]

    # Plot histograms of percentage errors
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(x_errors, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of X Coordinate Percentage Errors')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(y_errors, bins=20, color='salmon', edgecolor='black')
    plt.title('Histogram of Y Coordinate Percentage Errors')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('percentage_errors_histogram.png')  # Save the plot to a file
    plt.close()

    # Plot histograms of absolute errors
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(x_abs_errors, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of X Coordinate Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(y_abs_errors, bins=20, color='salmon', edgecolor='black')
    plt.title('Histogram of Y Coordinate Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('absolute_errors_histogram.png')  # Save the plot to a file
    plt.close()

    # Scatter plots of actual vs predicted values
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(sample_labels_rescaled[:, 0], predicted_labels_rescaled[:, 0], alpha=0.5)
    plt.title('Actual vs Predicted X Coordinates')
    plt.xlabel('Actual X')
    plt.ylabel('Predicted X')
    plt.plot([sample_labels_rescaled[:, 0].min(), sample_labels_rescaled[:, 0].max()], 
             [sample_labels_rescaled[:, 0].min(), sample_labels_rescaled[:, 0].max()], 'r--')

    plt.subplot(1, 2, 2)
    plt.scatter(sample_labels_rescaled[:, 1], predicted_labels_rescaled[:, 1], alpha=0.5)
    plt.title('Actual vs Predicted Y Coordinates')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.plot([sample_labels_rescaled[:, 1].min(), sample_labels_rescaled[:, 1].max()], 
             [sample_labels_rescaled[:, 1].min(), sample_labels_rescaled[:, 1].max()], 'r--')

    plt.tight_layout()
    plt.savefig('actual_vs_predicted_scatter.png')  # Save the plot to a file
    plt.close()

    # Calculate standard deviations
    std_actual_x = np.std(sample_labels_rescaled[:, 0])
    std_predicted_x = np.std(predicted_labels_rescaled[:, 0])
    std_actual_y = np.std(sample_labels_rescaled[:, 1])
    std_predicted_y = np.std(predicted_labels_rescaled[:, 1])

    # Plot distributions and calculate standard deviations
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(sample_labels_rescaled[:, 0], bins=30, alpha=0.5, label='Actual X', density=True)
    plt.hist(predicted_labels_rescaled[:, 0], bins=30, alpha=0.5, label='Predicted X', density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(sample_labels_rescaled[:, 0]), np.std(sample_labels_rescaled[:, 0]))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Distribution of X Coordinates\nStd actual: {std_actual_x:.2f}, Std predicted: {std_predicted_x:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(sample_labels_rescaled[:, 1], bins=30, alpha=0.5, label='Actual Y', density=True)
    plt.hist(predicted_labels_rescaled[:, 1], bins=30, alpha=0.5, label='Predicted Y', density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(sample_labels_rescaled[:, 1]), np.std(sample_labels_rescaled[:, 1]))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Distribution of Y Coordinates\nStd actual: {std_actual_y:.2f}, Std predicted: {std_predicted_y:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig('distributions.png')  # Save the plot to a file
    plt.close()