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
train_dataset, val_dataset = load_and_process_data(file_path, num_samples=10000, energy_threshold=500)

# Define the model builder for hyperparameter tuning
def model_builder(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(16, 16, 2)))
    
    # Hyperparameter search space for Conv2D layers
    for i in range(hp.Int('conv_blocks', 2, 5, default=3)):
        model.add(layers.Conv2D(
            filters=hp.Int('filters_' + str(i), 32, 256, step=32, default=64),
            kernel_size=hp.Choice('kernel_size_' + str(i), [3, 5, 7], default=3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.0001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(hp.Float('conv_dropout_' + str(i), 0.2, 0.5, step=0.1, default=0.3)))

    model.add(layers.Flatten())
    for i in range(hp.Int('dense_blocks', 1, 3, default=2)):
        model.add(layers.Dense(hp.Int('dense_units_' + str(i), 128, 512, step=64, default=256), activation='relu'))
        model.add(layers.Dropout(hp.Float('dense_dropout_' + str(i), 0.3, 0.7, step=0.1, default=0.5)))
    
    model.add(layers.Dense(2, kernel_regularizer=regularizers.l2(0.0001)))

    # Hyperparameter search space for optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse',
                  metrics=['mae'])

    return model

# Instantiate a tuner object
tuner = kt.Hyperband(
    model_builder,
    objective='val_mae',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='cnn_tuning'
)

# Build the initial model to print the summary
initial_model = model_builder(kt.HyperParameters())
initial_model.summary()

# Define early stopping and checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Run the hyperparameter search
tuner.search(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping, checkpoint])

# Debugging: Get the search summary
tuner.results_summary()

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('dense_units_0')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping, checkpoint])

# Function to plot training history
plot_training_history(history)
