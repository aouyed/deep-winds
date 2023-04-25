#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:24:06 2023

@author: aouyed
"""

import tensorflow as tf

# Define the input layers
input1 = tf.keras.layers.Input(shape=(None, None, None))
input2 = tf.keras.layers.Input(shape=(None, None, None))

# Concatenate the two input layers along the channel dimension
concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([input1, input2])
breakpoint()

# Define the rest of the model
x = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(concatenated_inputs)
x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
x = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)

# Define the output layers
output1 = tf.keras.layers.Dense(units=64, activation='softmax')(x)
output2 = tf.keras.layers.Dense(units=64, activation='softmax')(x)
output3 = tf.keras.layers.Dense(units=64, activation='softmax')(x)

# Define the model with inputs and outputs
model = tf.keras.models.Model(inputs=[input1, input2], outputs=[output1, output2, output3])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x=[input_data1, input_data2], y=[output_data1, output_data2, output_data3], epochs=10, batch_size=32)
