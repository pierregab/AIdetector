import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Add, Layer
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
import numpy as np
from Loading_data import load_and_process_data
from Plot_training import plot_training_history

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = Dropout(dropout)(attention_output)
    
    # Ensure the dimensions match for the residual connection
    if x.shape[-1] != inputs.shape[-1]:
        inputs = Dense(x.shape[-1])(inputs)
        
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = mlp(x, [ff_dim], dropout)
    if x.shape[-1] != res.shape[-1]:
        res = Dense(x.shape[-1])(res)
    return Add()([x, res])

class ViTHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        patch_size = hp.Int('patch_size', min_value=2, max_value=8, step=2)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        key_dim = hp.Choice('key_dim', values=[32, 64, 128])
        ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
        transformer_layers = hp.Int('transformer_layers', min_value=2, max_value=8, step=2)
        mlp_units = hp.Choice('mlp_units', values=[128, 256, 512])
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.3, step=0.1)

        inputs = layers.Input(shape=self.input_shape)
        # Create patches.
        patches = Patches(patch_size)(inputs)
        # Encode patches.
        num_patches = (self.input_shape[0] // patch_size) * (self.input_shape[1] // patch_size)
        encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=key_dim*num_heads)(patches)

        # Create multiple transformer blocks.
        for _ in range(transformer_layers):
            encoded_patches = transformer_encoder(encoded_patches, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout=dropout)

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(dropout)(representation)
        # Add MLP.
        features = mlp(representation, [mlp_units], dropout)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = tf.keras.Model(inputs=inputs, outputs=logits)
        
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='mse',
                      metrics=['mae'])
        
        return model

# Define the input shape and number of classes
input_shape = (16, 16, 2)
num_classes = 2

# Initialize the hypermodel
hypermodel = ViTHyperModel(input_shape=input_shape, num_classes=num_classes)

# Initialize the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_mae',
    max_trials=10,
    executions_per_trial=2,
    directory='vit_tuning',
    project_name='vision_transformer',
    overwrite=True  # Set overwrite=True to start a new search
)

# Load and preprocess the data
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

# Define early stopping and checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('vit_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Set verbose logging
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

# Run the tuner search
tuner.search(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[early_stopping, checkpoint], verbose=1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping, checkpoint])

# Plot the training history
plot_training_history(history)

# Save the final model
best_model.save('final_vit_model.keras')
