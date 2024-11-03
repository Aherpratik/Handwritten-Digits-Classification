# matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta

# TensorFlow version check
tf.__version__

# Load MNIST dataset using TensorFlow 2.x
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing the images to a [0, 1] scale
train_images = train_images / 255.0
test_images = test_images / 255.0

# Validation dataset
validation_images = train_images[50000:]
validation_labels = train_labels[50000:]
train_images = train_images[:50000]
train_labels = train_labels[:50000]

# One-hot encoding of labels
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
validation_labels = tf.keras.utils.to_categorical(validation_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

print("Size of:")
print("- Training-set:\t\t{}".format(len(train_labels)))
print("- Test-set:\t\t{}".format(len(test_labels)))
print("- Validation-set:\t{}".format(len(validation_labels)))

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128

# Convert images to have a single channel
train_images = train_images.reshape(-1, img_size, img_size, num_channels)
validation_images = validation_images.reshape(-1, img_size, img_size, num_channels)
test_images = test_images.reshape(-1, img_size, img_size, num_channels)

# Placeholders
x = tf.keras.layers.Input(shape=(img_size, img_size, num_channels), name='x')
y_true = tf.keras.layers.Input(shape=(num_classes,), name='y_true')

# Model building using Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=num_filters1, kernel_size=filter_size1, padding='same', activation='relu', input_shape=(img_size, img_size, num_channels)),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

    tf.keras.layers.Conv2D(filters=num_filters2, kernel_size=filter_size2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(fc_size, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(validation_images, validation_labels))

# Evaluating the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("\nTest set Accuracy: {:.1%}".format(test_accuracy))

# Function to plot example errors
def plot_example_errors(images, cls_true, cls_pred):
    incorrect = (cls_true != cls_pred)

    images = images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    plot_images(images=images[:9], cls_true=cls_true[:9], cls_pred=cls_pred[:9])

# Predicting on the test set
y_pred = model.predict(test_images)
cls_pred = np.argmax(y_pred, axis=1)
cls_true = np.argmax(test_labels, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(cls_true, cls_pred)
print(conf_matrix)

plt.matshow(conf_matrix)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Accuracy on test set
correct = (cls_true == cls_pred)
correct_sum = correct.sum()
num_test = len(test_images)

acc = float(correct_sum) / num_test
print("\nAccuracy on Test-Set: {0:.1%} ({1} / {2})".format(acc, correct_sum, num_test))