import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Add, Layer
from tensorflow.keras.optimizers import Adam
import numpy as np
from Loading_data import load_and_process_data
from Plot_training import plot_training_history

# Helper function to create the Vision Transformer model
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = Dropout(dropout_rate)(x)
    return x

def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = Dropout(dropout)(attention_output)
    
    # Ensure the dimensions match for the residual connection
    if x.shape[-1] != inputs.shape[-1]:
        inputs = Dense(x.shape[-1], kernel_regularizer=regularizers.l2(1e-4))(inputs)
        
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = mlp(x, [ff_dim], dropout)
    if x.shape[-1] != res.shape[-1]:
        res = Dense(x.shape[-1], kernel_regularizer=regularizers.l2(1e-4))(res)
    return Add()([x, res])

def build_vit_model(input_shape, num_classes, patch_size=4, transformer_layers=6, 
                    num_heads=6, key_dim=64, ff_dim=128, mlp_units=[256], dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=key_dim*num_heads)(patches)

    # Create multiple transformer blocks.
    for _ in range(transformer_layers):
        encoded_patches = transformer_encoder(encoded_patches, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout=dropout)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout)(representation)
    # Add MLP.
    features = mlp(representation, mlp_units, dropout)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation='sigmoid')(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

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

input_shape = (16, 16, 2)
num_classes = 2
vit_model = build_vit_model(input_shape, num_classes)

# Compile the model
vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss='mse',
                  metrics=['mae'])

# Display the model summary
vit_model.summary()

file_path = 'H01_labelCNN_50x50grid_RAWPM.bin'

# Assuming load_and_process_data is a predefined function
train_dataset, val_dataset = load_and_process_data(file_path, use_real_positions=True, num_samples=100000, test_size=0.2, batch_size=128)

# Define early stopping and checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'vit_model.h5',  # Use .h5 extension to specify HDF5 format
    save_best_only=True, 
    monitor='val_loss', 
    save_weights_only=False,  
    save_format='h5'  # Explicitly set the save format to h5
)

# Train the model
history = vit_model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping, checkpoint])

# Plot the training history (assuming plot_training_history function is defined)
plot_training_history(history)

# Save the model at the end of training
vit_model.save('final_vit_model.h5', save_format='h5')
