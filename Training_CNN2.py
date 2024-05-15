import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from Loading_data import load_and_process_data
from Plot_training import plot_training_history

def build_model(hp):
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(16, 16, 2)))
    
    # First Conv Block
    model.add(layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv_1_kernel', values=[3,5]),
        activation='relu',
        padding='same'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=hp.Float('conv_1_dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Second Conv Block
    model.add(layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=256, step=64),
        kernel_size=hp.Choice('conv_2_kernel', values=[3,5]),
        activation='relu',
        padding='same'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=hp.Float('conv_2_dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Third Conv Block
    model.add(layers.Conv2D(
        filters=hp.Int('conv_3_filter', min_value=128, max_value=512, step=128),
        kernel_size=hp.Choice('conv_3_kernel', values=[3,5]),
        activation='relu',
        padding='same'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=hp.Float('conv_3_dropout', min_value=0.2, max_value=0.5, step=0.1)))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'))
    model.add(layers.Dropout(rate=hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layers.Dense(2, activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mse',
        metrics=['mae']
    )

    return model

# Instantiate the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=20,
    factor=3,
    directory='keras_tuner_dir',
    project_name='arrow_orientation'
)

# Load and process the data
file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
train_dataset, val_dataset = load_and_process_data(file_path, use_real_positions=True, num_samples=100000, test_size=0.2, batch_size=64)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
val_dataset = val_dataset.map(lambda x, y: (layers.Rescaling(1./255)(x), y))

# Perform hyperparameter search
tuner.search(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10)])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of the best model
best_model.summary()

# Train the best model further if needed
history = best_model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10)])

# Plot the training history
plot_training_history(history)

# Save the final model
best_model.save('best_cnn_model.keras')
