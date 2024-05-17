import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Plot_training import plot_training_history
from Loading_data import load_and_process_data

# Define the model with the best hyperparameters found
def build_best_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(16, 16, 2)))
    
    # Adding convolutional blocks
    model.add(layers.Conv2D(192, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))  # Reduced dropout

    model.add(layers.Conv2D(224, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))  # Reduced dropout

    model.add(layers.Conv2D(32, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))  # Reduced dropout

    model.add(layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.2))  # Reduced dropout

    model.add(layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.2))  # Reduced dropout

    # Flatten the output and add dense layers
    model.add(layers.Flatten())

    model.add(layers.Dense(320, activation='relu'))
    model.add(layers.Dropout(0.3))  # Reduced dropout

    model.add(layers.Dense(320, activation='relu'))
    model.add(layers.Dropout(0.3))  # Reduced dropout

    model.add(layers.Dense(448, activation='relu'))
    model.add(layers.Dropout(0.2))  # Reduced dropout

    model.add(layers.Dense(2, activation='sigmoid'))  # Sigmoid activation for [0, 1] output

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mse',  # MSE is suitable for regression even with sigmoid
                  metrics=['mae'])

    return model

# Build the best model and print the summary
best_model = build_best_model()
best_model.summary()

# Define early stopping and checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
train_dataset, val_dataset = load_and_process_data(file_path, num_samples=1000000, use_real_positions=True)
history = best_model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[early_stopping, checkpoint])

# Save the model
best_model.save('final_model.keras')

# Function to plot training history
plot_training_history(history)