import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from scipy.stats import norm

def test_model(model_path, data_path, num_samples=1000, grid_size=16):
    # Load the entire data from the binary file
    dtype_here = np.dtype([
        ('charge', np.float32, (256,)),
        ('time', np.int32, (256,)),
        ('positions_ml', np.float32, (2,)),
        ('positions_real', np.float32, (2,)),
        ('energy', np.float32)
    ])

    data = np.fromfile(data_path, dtype=dtype_here)

    # Reshape the charge and time data to 16x16 images
    charge_data = data['charge'].reshape((-1, 16, 16))
    time_data = data['time'].reshape((-1, 16, 16)).astype(np.float32)  # Ensure time data is in float

    # Extract other data
    positions_ml = data['positions_ml']
    positions_real = data['positions_real']
    energy = data['energy']

    # Sample random indices
    total_samples = charge_data.shape[0]
    indices = np.random.choice(total_samples, num_samples, replace=False)

    # Subset the data using the sampled indices
    charge_data_sampled = charge_data[indices]
    time_data_sampled = time_data[indices]
    positions_ml_sampled = positions_ml[indices]
    positions_real_sampled = positions_real[indices]

    # Normalize the sampled data to fit within 0 to 1
    charge_data_sampled = charge_data_sampled / np.max(charge_data_sampled)
    time_data_sampled = time_data_sampled / np.max(time_data_sampled)
    positions_real_sampled = (positions_real_sampled - positions_real_sampled.min()) / (positions_real_sampled.max() - positions_real_sampled.min())

    # Stack charge and time data to create a (16, 16, 2) input for the model
    model_input_sampled = np.stack((charge_data_sampled, time_data_sampled), axis=-1)

    # Load the Keras model
    model = load_model(model_path)

    # Generate predictions
    predictions_sampled = model.predict(model_input_sampled)

    # Print 100 first predictions against real positions
    print('Predictions vs Real Positions:')
    for i in range(100):
        print(f'Prediction: {predictions_sampled[i]}, Real: {positions_real_sampled[i]}')

    # Calculate the position differences
    position_diff_x = predictions_sampled[:, 0] - positions_real_sampled[:, 0]
    position_diff_y = predictions_sampled[:, 1] - positions_real_sampled[:, 1]

    # Denormalize the positions
    position_diff_x = position_diff_x * grid_size
    position_diff_y = position_diff_y * grid_size

    # Calculate mean and standard deviation
    mean_diff_x = np.mean(position_diff_x)
    std_diff_x = np.std(position_diff_x)
    mean_diff_y = np.mean(position_diff_y)
    std_diff_y = np.std(position_diff_y)

    # Calculate the normalized MAE (Mean Absolute Error)
    norm_positions_real = np.clip(positions_real_sampled / grid_size, 0, 1)
    norm_predictions = np.clip(predictions_sampled, 0, 1)

    mae_x = np.mean(np.abs(norm_predictions[:, 0] - norm_positions_real[:, 0]))
    mae_y = np.mean(np.abs(norm_predictions[:, 1] - norm_positions_real[:, 1]))

    # Plot the distributions with KDE
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # X position difference
    sns.histplot(position_diff_x, bins=50, color='cyan', kde=True, ax=axs[0], stat='density')
    xmin, xmax = axs[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p_x = norm.pdf(x, mean_diff_x, std_diff_x)
    axs[0].plot(x, p_x, 'k', linewidth=2, label='Gaussian fit')
    axs[0].set_title('X Position Difference')
    axs[0].set_xlabel('Difference')
    axs[0].set_ylabel('Density')
    axs[0].annotate(f'Std: {std_diff_x:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', ha='left', va='top')
    axs[0].annotate(f'Norm MAE: {mae_x:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', ha='left', va='top')
    axs[0].legend(loc='upper left')

    # Y position difference
    sns.histplot(position_diff_y, bins=50, color='magenta', kde=True, ax=axs[1], stat='density')
    xmin, xmax = axs[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p_y = norm.pdf(x, mean_diff_y, std_diff_y)
    axs[1].plot(x, p_y, 'k', linewidth=2, label='Gaussian fit')
    axs[1].set_title('Y Position Difference')
    axs[1].set_xlabel('Difference')
    axs[1].set_ylabel('Density')
    axs[1].annotate(f'Std: {std_diff_y:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', ha='left', va='top')
    axs[1].annotate(f'Norm MAE: {mae_y:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', ha='left', va='top')
    axs[1].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('position_difference_with_kde_keras_sampled.png')  # Save the plot to a file
    plt.show()

    # Scatter plot of real vs computed values for positions with density visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Denormalize the positions
    positions_real_sampled = positions_real_sampled * grid_size
    predictions_sampled = predictions_sampled * grid_size

    # Hexbin plot for X coordinates
    hb_x = axs[0].hexbin(positions_real_sampled[:, 0], predictions_sampled[:, 0], gridsize=50, cmap='Blues', mincnt=1)
    axs[0].set_title('Actual vs Predicted X Coordinates')
    axs[0].set_xlabel('Actual X')
    axs[0].set_ylabel('Predicted X')
    axs[0].plot([positions_real_sampled[:, 0].min(), positions_real_sampled[:, 0].max()], 
                [positions_real_sampled[:, 0].min(), positions_real_sampled[:, 0].max()], 'r--')
    cb = fig.colorbar(hb_x, ax=axs[0], label='Counts')

    # Hexbin plot for Y coordinates
    hb_y = axs[1].hexbin(positions_real_sampled[:, 1], predictions_sampled[:, 1], gridsize=50, cmap='Greens', mincnt=1)
    axs[1].set_title('Actual vs Predicted Y Coordinates')
    axs[1].set_xlabel('Actual Y')
    axs[1].set_ylabel('Predicted Y')
    axs[1].plot([positions_real_sampled[:, 1].min(), positions_real_sampled[:, 1].max()], 
                [positions_real_sampled[:, 1].min(), positions_real_sampled[:, 1].max()], 'r--')
    cb = fig.colorbar(hb_y, ax=axs[1], label='Counts')

    plt.tight_layout()
    plt.savefig('actual_vs_predicted_hexbin.png')  # Save the plot to a file
    plt.show()

    # Heatmap of average and absolute average position differences in grid cells
    avg_diff_grid_x, avg_diff_grid_y, abs_avg_diff_grid_x, abs_avg_diff_grid_y = calculate_position_diff_grids(positions_real_sampled, predictions_sampled, grid_size)

    # Plot the heatmaps for average differences
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    sns.heatmap(avg_diff_grid_x, ax=axs[0], cmap='viridis', cbar_kws={'label': 'Average Position Difference X'})
    axs[0].set_title('Heatmap of Average Position Difference (X)')
    axs[0].set_xlabel('Grid X')
    axs[0].set_ylabel('Grid Y')
    axs[0].set_aspect('equal', adjustable='box')  # Ensure the heatmap is squared

    sns.heatmap(avg_diff_grid_y, ax=axs[1], cmap='viridis', cbar_kws={'label': 'Average Position Difference Y'})
    axs[1].set_title('Heatmap of Average Position Difference (Y)')
    axs[1].set_xlabel('Grid X')
    axs[1].set_ylabel('Grid Y')
    axs[1].set_aspect('equal', adjustable='box')  # Ensure the heatmap is squared

    plt.tight_layout()
    plt.savefig('average_position_difference_heatmap.png')  # Save the plot to a file
    plt.show()

    # Plot the heatmaps for absolute average differences
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    sns.heatmap(abs_avg_diff_grid_x, ax=axs[0], cmap='viridis', cbar_kws={'label': 'Absolute Average Position Difference X'})
    axs[0].set_title('Heatmap of Absolute Average Position Difference (X)')
    axs[0].set_xlabel('Grid X')
    axs[0].set_ylabel('Grid Y')
    axs[0].set_aspect('equal', adjustable='box')  # Ensure the heatmap is squared

    sns.heatmap(abs_avg_diff_grid_y, ax=axs[1], cmap='viridis', cbar_kws={'label': 'Absolute Average Position Difference Y'})
    axs[1].set_title('Heatmap of Absolute Average Position Difference (Y)')
    axs[1].set_xlabel('Grid X')
    axs[1].set_ylabel('Grid Y')
    axs[1].set_aspect('equal', adjustable='box')  # Ensure the heatmap is squared

    plt.tight_layout()
    plt.savefig('absolute_average_position_difference_heatmap.png')  # Save the plot to a file
    plt.show()


def calculate_position_diff_grids(positions_real_sampled, predictions_sampled, grid_size=16):
    # Normalize positions to the range [0, 1)
    positions_real_normalized = positions_real_sampled / np.max(positions_real_sampled, axis=0)

    # Calculate the pixel differences
    position_diff_x = predictions_sampled[:, 0] - positions_real_sampled[:, 0]
    position_diff_y = predictions_sampled[:, 1] - positions_real_sampled[:, 1]
    abs_position_diff_x = np.abs(position_diff_x)
    abs_position_diff_y = np.abs(position_diff_y)

    # Initialize the error grids
    avg_diff_grid_x = np.zeros((grid_size, grid_size))
    avg_diff_grid_y = np.zeros((grid_size, grid_size))
    abs_avg_diff_grid_x = np.zeros((grid_size, grid_size))
    abs_avg_diff_grid_y = np.zeros((grid_size, grid_size))
    counts_grid = np.zeros((grid_size, grid_size))

    # Determine which real positions fall into each pixel
    x_idx = (positions_real_normalized[:, 0] * grid_size).astype(int)
    y_idx = (positions_real_normalized[:, 1] * grid_size).astype(int)

    # Ensure indices are within bounds
    x_idx = np.clip(x_idx, 0, grid_size - 1)
    y_idx = np.clip(y_idx, 0, grid_size - 1)

    # Accumulate the differences in each grid cell
    for i in range(len(positions_real_sampled)):
        xi = x_idx[i]
        yi = y_idx[i]
        avg_diff_grid_x[yi, xi] += position_diff_x[i]
        avg_diff_grid_y[yi, xi] += position_diff_y[i]
        abs_avg_diff_grid_x[yi, xi] += abs_position_diff_x[i]
        abs_avg_diff_grid_y[yi, xi] += abs_position_diff_y[i]
        counts_grid[yi, xi] += 1

    # Avoid division by zero
    counts_grid[counts_grid == 0] = 1

    # Calculate the average differences in each grid cell
    avg_diff_grid_x /= counts_grid
    avg_diff_grid_y /= counts_grid
    abs_avg_diff_grid_x /= counts_grid
    abs_avg_diff_grid_y /= counts_grid

    return avg_diff_grid_x, avg_diff_grid_y, abs_avg_diff_grid_x, abs_avg_diff_grid_y
