from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))  # 200, 200, 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# print(x_train.shape)
# print(x_test.shape)

def run_test_harness():
    model = models.Sequential()
    # model.add(layers.UpSampling2D((2,2)))
    # model.add(layers.UpSampling2D((2,2)))
    # model.add(layers.UpSampling2D((2,2)))
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # prepare iterator
    it_train = datagen.flow(x_train, y_train, batch_size=64)
    # fit model
    steps = int(x_train.shape[0] / 64)
    history = model.fit(it_train, steps_per_epoch=steps, epochs=50, validation_data=(x_test, y_test), verbose=1)

    model.save('resnet50.h5')

    # model.evaluate(x_test, y_test)


def resnet_prediction():
    model = tf.keras.models.load_model('resnet50.h5')

    start = time.time()
    for i in range(100):
        ran_img = np.random.randint(low=0, high=1000)
        result = model.predict_classes(x_test[ran_img].reshape(1, 32, 32, 3))
        print(result)
    end = time.time()

    avg_time = (end - start) / 100
    print(avg_time)  # 0.04398 #0.04471 #0.044018


if __name__ == '__main__':
    # run_test_harness()
    resnet_prediction()
