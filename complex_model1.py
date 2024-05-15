import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from Loading_data import load_and_process_data
from Plot_training import plot_training_history
import numpy as np

# Load and process the data
file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'
train_dataset, val_dataset = load_and_process_data(file_path, use_real_positions=False, num_samples=100000, test_size=0.2, batch_size=64)

# Define a dense block
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        cb = layers.BatchNormalization()(x)
        cb = layers.ReLU()(cb)
        cb = layers.Conv2D(growth_rate, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.0001))(cb)
        x = layers.Concatenate()([x, cb])
    return x

# Define a transition block
def transition_block(x, reduction):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(int(x.shape[-1] * reduction), kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.AvgPool2D(pool_size=2, strides=2)(x)
    return x

# Define the more advanced DenseNet-like model
def build_densenet_model():
    inputs = layers.Input(shape=(16, 16, 2))
    
    # Initial Conv Layer
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Dense Blocks with Transition Layers
    x = dense_block(x, num_layers=4, growth_rate=32)
    x = transition_block(x, reduction=0.5)
    x = dense_block(x, num_layers=4, growth_rate=32)
    x = transition_block(x, reduction=0.5)
    x = dense_block(x, num_layers=4, growth_rate=32)
    
    # Global Average Pooling and Dense Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, kernel_regularizer=regularizers.l2(0.0001))(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='mse',
                  metrics=['mae'])
    
    return model

# Build the DenseNet-like model
densenet_model = build_densenet_model()

# Display the model summary
densenet_model.summary()

# Define early stopping and checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('densenet_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the DenseNet-like model
history = densenet_model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping, checkpoint])

# Plot the training history
plot_training_history(history)

# Save the model at the end of training
densenet_model.save('final_densenet_model.keras')
