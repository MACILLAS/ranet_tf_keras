"""
Main File for RaNet on TF 2
"""
import hyperparameters as hp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Lambda, ReLU, Input, UpSampling2D
from tensorflow.keras.models import Model

from models.small_net import SmallModel
from models.med_net import MedModel

def train(model, X_train, y_train, X_test, y_test, checkpoint_path,datagen =None):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True)
    ]

    # Begin training
    if datagen is None:
        model.fit(
            x=X_train, y=y_train,
            epochs=hp.num_epochs,
            batch_size=hp.batch_size,
            callbacks=callback_list,
            validation_data=(X_test, y_test)
            )
    else:
        model.fit(
            datagen.flow(X_train, y_train, batch_size=hp.batch_size),
            epochs=hp.num_epochs,            
            callbacks=callback_list,
            validation_data=(X_test, y_test)
            )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def main():

    # Setting class names for the dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Loading the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize images
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # large data
    X_train_large = X_train
    X_test_large = X_test

    # medium data
    X_train_med = tf.image.resize_with_pad(X_train, 16, 16, antialias=True)
    X_test_med = tf.image.resize_with_pad(X_test, 16, 16, antialias=True)

    # small data
    X_train_small = tf.image.resize_with_pad(X_train, 8, 8, antialias=True)
    X_test_small = tf.image.resize_with_pad(X_test, 8, 8, antialias=True)

    ### Create Instance of SmallModel
    small_model = SmallModel()

    small_model(tf.keras.Input(shape=(8, 8, 3)))
    checkpoint_path = "./small_model_checkpoints/"
    small_model.summary()

    # Compile model graph
    small_model.compile(
        optimizer=small_model.optimizer,
        loss=small_model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    # Create checkpoint path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # train(small_model, X_train_small, y_train, X_test_small, y_test, checkpoint_path)

    ### Create Instance of MedModel
    med_model = MedModel()

    med_model(tf.keras.Input(shape=(16, 16, 3)))
    checkpoint_path = "./med_model_checkpoints/"
    med_model.summary()

    # Compile model graph
    med_model.compile(
        optimizer=med_model.optimizer,
        loss=med_model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    # Create checkpoint path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    train(med_model, X_train_med, y_train, X_test_med, y_test, checkpoint_path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
