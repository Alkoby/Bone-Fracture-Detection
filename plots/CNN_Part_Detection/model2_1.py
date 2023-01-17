import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os
import cv2
import random
import pickle
from tensorflow.keras.callbacks import TensorBoard

datadir = "C:/Users/Naor0/PycharmProjects/Bone-Fracture-Detection/PartData"
# 0 - Elbow , 1 - Forearm , 2 - Humerus , 3 - Shoulder
categories = ["Elbow", "Forearm", "Humerus", "Shoulder"]

for category in categories:
    path = os.path.join(datadir, category)  # path for each category folder
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img_array, cmap="gray")
        # plt.show()
        # break
    print(img_array.shape)
    img_size = 50
    new_array = cv2.resize(img_array, (img_size, img_size))
    # plt.imshow(new_array, cmap='gray')
    # plt.show()

training_data = []


def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)  # path for each category folder
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
print(len(training_data))

random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])

x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)
x = np.array(x).reshape(-1, img_size, img_size, 1)

pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("x.pickle", "rb")
x = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

x = x / 255.0

NAME = "Bones Types"
model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

#early_stop = EarlyStopping(monitor="accuracy", mode="min", patience=5, restore_best_weights=True)
y = np.array(y)
history = model.fit(x, y,
          batch_size=32,
          epochs=12,
          validation_split=0.3)

model.evaluate(x,y)
model.save("Body_Part_Detection.h5")
pred = model.predict(x)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
