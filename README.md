# Handwritten-Text-Recognition-
# Handwritten Text Recognition with TensorFlow
pip install tensorflow numpy matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to a range of 0 to 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images
model.add(Dense(128, activation='relu'))   # Hidden layer with 128 neurons
model.add(Dropout(0.2))                    # Dropout layer for regularization
model.add(Dense(10, activation='softmax')) # Output layer with 10 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Function to predict a single image
def predict_image(image):
    image = image.reshape(1, 28, 28)  # Reshape for the model
    prediction = model.predict(image)
    return np.argmax(prediction)

# Test the model with a sample image
sample_image = x_test[0]  # Take the first test image
plt.imshow(sample_image, cmap='gray')
plt.title(f'Predicted: {predict_image(sample_image)}')
plt.show()
