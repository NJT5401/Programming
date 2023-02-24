import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# load the dataset
fashion_mnist = keras.datasets.fashion_mnist

# pull data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model_file = 'model.h5'


# print(train_labels)
# plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
# plt.show()

if os.path.exists(model_file):
    model = load_model(model_file)
else:

    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(28,28)),
        
        # first hidden layer
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        
        # output layer
        keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    # compiles the model to an actual neural net. Loss function tells us how wrong we are, optimizer tweaks weights accordingly
    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

    # provide images and labels to the training. Epochs determines how many times it runs through the dataset and adjusts the weights
    # increasing epochs does decrease loss but it has a plateau proportional to the complexity of the data and neural net
    model.fit(train_images, train_labels, epochs=15)
    
    model.save(model_file)

# testing the model
print("Enter image 0 - 9999: ")
test_image = int(input())
while test_image >= 0 and test_image <= 9999:
    # test_loss = model.evaluate(test_images, test_labels)

    # make predictions
    predictions = model.predict(test_images)

    # dictionary to translate index to word
    label_dict = {0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

    # prints neural net's guess
    index = list(predictions[test_image]).index(max(predictions[test_image]))
    print(f'Guess: {label_dict[index]}')

    # prints correct answer
    correct_index = test_labels[test_image]
    print(f'Answer: {label_dict[correct_index]}')

    # shows inputted image
    plt.imshow(test_images[test_image], cmap='gray', vmin=0, vmax=255)
    plt.show()

    print("Enter image 0 - 9999: ")
    test_image = int(input())