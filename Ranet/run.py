"""
Main File for RaNet on TF 2
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import hyperparameters as hp
import datetime
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import time

from models.ranet import FullRanet

def scheduler(epoch, lr):
    lr1 = 0.0001
    lr2 = 0.00001
    lr3 = 0.000005
    if epoch < 60:
        return lr1
    elif epoch < 100:
        return lr2
    else:
        return lr3

#def four_way(datagen, x, y):
#    iterator = datagen.flow(x, y, batch_size=hp.batch_size)
#    yield iterator.x, {"output_1": iterator.y, "output_2": iterator.y, "output_3": iterator.y}

def train(model, X_train, y_train, X_test, y_test, checkpoint_path):
    """ Training routine. """

    datagen = ImageDataGenerator(
        samplewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # Keras callbacks for training
    callback_list = [
        tensorboard_callback,
        LearningRateScheduler(scheduler),
        ModelCheckpoint(
            filepath=checkpoint_path + \
                     "weights.e{epoch:02d}.h5",
            monitor='val_output_3_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True)
    ]

    # #Begin training
    # history = model.fit(x=X_train, y={"output_1": y_train, "output_2": y_train, "output_3": y_train},
    #                     validation_data=(X_test, {"output_1": y_test, "output_2": y_test, "output_3": y_test}),
    #                     epochs=hp.num_epochs, verbose=1, batch_size=hp.batch_size, validation_batch_size=hp.batch_size,
    #                     callbacks=callback_list)

    model.fit(datagen.flow(X_train, y_train), steps_per_epoch=len(X_train) // hp.batch_size,
                        validation_data=(X_test, {"output_1": y_test, "output_2": y_test, "output_3": y_test}),
                        validation_steps=len(X_test) // hp.batch_size,
                        epochs=hp.num_epochs, verbose=1, callbacks=callback_list)

    return model

def test(model, test_data):
    """ Testing routine. """
    # Run model on test set
    pred, conf = model.predict(x=test_data[0].reshape(1, 32, 32, 3))


def main():

    # Setting class names for the dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Loading the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize images
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = FullRanet()

    init_lr = hp.learning_rate
    epochs = hp.num_epochs

    model.compile(optimizer=SGD(momentum=0.9, lr=init_lr),
                  loss={
                      'output_1': 'sparse_categorical_crossentropy',
                      'output_2': 'sparse_categorical_crossentropy',
                      'output_3': 'sparse_categorical_crossentropy'},
                  loss_weights={
                      'output_1': 0.5,
                      'output_2': 1,
                      'output_3': 1.5},
                  metrics={
                      'output_1': 'sparse_categorical_accuracy',
                      'output_2': 'sparse_categorical_accuracy',
                      'output_3': 'sparse_categorical_accuracy'})

    checkpoint_path = "./model_checkpoints/"

    # Create checkpoint path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model = train(model, X_train, y_train, X_test, y_test, checkpoint_path)
    model.save_weights("ranet_model.h5")


def ranet_prediction():
    model = FullRanet()
    # Loading the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # normalize images
    X_train = X_train / 255.0
    #model = model.build(input_shapes=(32, 32, 3))

    #what if we train 1 epoch then load_weights?
    model.compile(optimizer=SGD(momentum=0.9, lr=0.001),
                  loss={
                      'output_1': 'sparse_categorical_crossentropy',
                      'output_2': 'sparse_categorical_crossentropy',
                      'output_3': 'sparse_categorical_crossentropy'},
                  loss_weights={
                      'output_1': 0.5,
                      'output_2': 1,
                      'output_3': 1.5},
                  metrics={
                      'output_1': 'sparse_categorical_accuracy',
                      'output_2': 'sparse_categorical_accuracy',
                      'output_3': 'sparse_categorical_accuracy'})

    model.fit(x=X_train, y={"output_1": y_train, "output_2": y_train, "output_3": y_train}, epochs=1, verbose=1, batch_size=hp.batch_size)

    model.load_weights("./ranet_model.h5")

    imgs = []
    gty = []
    for i in range(100):
        ran_img = np.random.randint(low=0, high=1000)
        img = tf.image.convert_image_dtype(X_test[ran_img].reshape(1, 32, 32, 3), dtype=tf.float32)
        imgs.append(img)
        temp = y_test[ran_img]
        gty.append(temp[0])

    start = time.time()
    predy = []
    for img in imgs:
        #img = X_test[0].reshape(1, 32, 32, 3)
        #img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        pred, conf = model.predict(img)
        predy.append(pred)
    end = time.time()

    gty = np.array(gty)
    predy = np.array(predy)
    diff = np.subtract(gty, predy)
    acc = np.count_nonzero(diff==0)

    comp_time = (end - start)/100
    print(comp_time)#0.017 (85%) #0.0162 (89%) #0.0157 (87%)
    print(acc)

if __name__ == '__main__':
    #main()
    ranet_prediction()
