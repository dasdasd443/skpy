import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

mnist = tf.keras.datasets.mnist #sample 28x28 datasets of handwritten 0-9 digits
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)
IMG_SIZE = 100
training_data = []

def load_data(): 
    dir = "C:\\Users\\cjvcr\\Desktop\\Personal\\skpy\\PetImages"
    categories = ["Cat","Dog"]

    for category in categories:
        path = os.path.join(dir, category)
        category_idx = categories.index(category)
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_img, category_idx])
            except Exception as e:
                pass
    
    random.shuffle(training_data)

def train():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    model.save('number_prediction.model')

def predict(idx = 0):
    model = tf.keras.models.load_model('number_prediction.model')
    prediction = model.predict([x_test])
    print(np.argmax(prediction[idx]))
    plt.imshow(x_test[idx])
    plt.show()

choice = input("Enter your choice: \nA. Train Model\nB: Predict\nC: Load Data\n")

if(choice == 'A'): 
    (x_train, x_test) = train()
elif(choice == 'B'):
    idx = input("Enter prediction index: ")
    predict(int(idx))
elif(choice == 'C'):
    load_data()