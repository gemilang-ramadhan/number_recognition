import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt

dataset_path = './dataset'

# Load training dataset
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,  # Reserve 20% of data for validation
    subset="training",
    seed=123,
    image_size=(90, 140),  # Adjust this size according to your dataset
    color_mode='grayscale',  # Since MNIST is grayscale
    batch_size=32
)

# Load validation dataset
validation_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(90, 140),
    color_mode='grayscale',
    batch_size=32
)

# Normalize the images
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize_img)
validation_dataset = validation_dataset.map(normalize_img)

# Optimize the dataset pipeline
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Build the Convolutional Neural Network (CNN) model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(90, 140, 1)),  # Correct input shape
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Reduce learning rate if necessary
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and store the history
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5  # Reduce number of epochs to decrease training time
)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_dataset, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Plot training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
