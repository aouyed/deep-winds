from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def MyNet(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Define convolutional layers
    x = layers.Conv3D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv3D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(x)

    # Define upsampling layers
    x = layers.UpSampling3D(size=2)(x)

    # Define final convolutional layers
    x = layers.Conv3D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv3D(16, kernel_size=3, padding='same', activation='relu')(x)
    outputs = layers.Conv3D(3, kernel_size=3, padding='same', activation='relu')(x)

    # Define model
    model = keras.Model(inputs=inputs, outputs=outputs, name='MyNet')
    return model


model = MyNet(input_shape=(32, 32, 32, 2))

# Create random input volume with 2 channels
input_volume = np.random.randn(1, 32, 32, 32, 2)

# Pass input volume through network
output_volume = model.predict(input_volume)

# Print output shape
print(output_volume)