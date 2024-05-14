import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from Loading_data import load_and_process_data
from Plot_training import plot_training_history
from sklearn.model_selection import train_test_split
import numpy as np

# Load and process the data
file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
train_data, train_labels = load_and_process_data(file_path, num_samples=1000000, energy_threshold=500)

# Debugging: Print data shapes
print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)

# Make sure data is scaled appropriately
train_data = train_data / np.max(train_data)  # Normalize by max value
train_labels = train_labels / 16 if train_labels.max() > 1 else train_labels  # Normalize only if necessary

# Debugging: Print max values to verify normalization
print("Max value in train data:", np.max(train_data))
print("Max value in train labels:", np.max(train_labels))

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Debugging: Print split data shapes
print("Train data shape after split:", train_data.shape)
print("Validation data shape after split:", val_data.shape)
print("Train labels shape after split:", train_labels.shape)
print("Validation labels shape after split:", val_labels.shape)

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the model with the best hyperparameters
def build_best_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(16, 16, 2)))
    
    # First Conv Block
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Conv Block
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Conv Block
    model.add(layers.Conv2D(
        filters=96,
        kernel_size=5,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, kernel_regularizer=regularizers.l2(0.0001)))

    # Compile the model with the best learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse',
                  metrics=['mae'])

    return model

# Build the model
best_model = build_best_model()

# Display the model summary
best_model.summary()

# Define early stopping and checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model with the best hyperparameters
history = best_model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[early_stopping, checkpoint])

# Plot the training history (assuming plot_training_history function is defined)
plot_training_history(history)

# Save the model at the end of training
best_model.save('final_model.keras')
