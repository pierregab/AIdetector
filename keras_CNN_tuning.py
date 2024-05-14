import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from Loading_data import load_and_process_data
from Plot_training import plot_training_history
import numpy as np

# Load and process the data
file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
train_data, train_labels = load_and_process_data(file_path, num_samples=1000000, energy_threshold=500)

# Make sure data is scaled appropriately
train_data = train_data / np.max(train_data)  # Normalize by max value
train_labels = train_labels / 16 if train_labels.max() > 1 else train_labels  # Normalize only if necessary

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

# Data Augmentation function
def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.rot90(image, k=np.random.randint(4))  # Random 90-degree rotation
    return image, label

# Define the model builder for hyperparameter tuning
def model_builder(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(16, 16, 2)))
    
    # Hyperparameter search space for Conv2D layers
    for i in range(hp.Int('conv_blocks', 1, 3, default=2)):
        model.add(layers.Conv2D(
            filters=hp.Int('filters_' + str(i), 32, 128, step=32, default=64),
            kernel_size=hp.Choice('kernel_size_' + str(i), [3, 5], default=3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('dense_units', 64, 256, step=64, default=128), activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_rate', 0.3, 0.7, step=0.1, default=0.5)))
    model.add(layers.Dense(2, kernel_regularizer=regularizers.l2(0.0001)))

    # Hyperparameter search space for optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse',
                  metrics=['mae'])

    return model

# Apply data augmentation to the training dataset
train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Set up Keras Tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_mae',
                     max_epochs=30,
                     factor=3,
                     directory='my_dir',
                     project_name='cnn_tuning')

# Define early stopping and checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Run the hyperparameter search
tuner.search(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[early_stopping, checkpoint])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('dense_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[early_stopping, checkpoint])

# Function to plot training history (assuming it's defined somewhere)
plot_training_history(history)
