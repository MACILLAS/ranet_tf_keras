"""
Main File for RaNet on TF 2
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import hyperparameters as hp

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from models.ranet import FullRanet

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    elif epoch < 250:
        return lr * 0.1
    else:
        return lr * 0.01

def train(model, X_train, y_train, X_test, y_test, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        LearningRateScheduler(scheduler),
        ModelCheckpoint(
            filepath=checkpoint_path + \
                     "weights.e{epoch:02d}.h5",
            monitor='val_output_1_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True)
    ]

    # Begin training
    history = model.fit(x=X_train, y={"output_1": y_train, "output_2": y_train, "output_3": y_train},
                        validation_data=(X_test, {"output_1": y_test, "output_2": y_test, "output_3": y_test}),
                        epochs=hp.num_epochs, verbose=1, batch_size=hp.batch_size, validation_batch_size=hp.batch_size,
                        callbacks=callback_list)

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

    model.compile(optimizer=SGD(momentum=0.9, lr=init_lr, decay=init_lr / epochs),
                  loss={
                      'output_1': 'sparse_categorical_crossentropy',
                      'output_2': 'sparse_categorical_crossentropy',
                      'output_3': 'sparse_categorical_crossentropy'},
                  loss_weights={
                      'output_1': 1,
                      'output_2': 1,
                      'output_3': 1},
                  metrics={
                      'output_1': 'sparse_categorical_accuracy',
                      'output_2': 'sparse_categorical_accuracy',
                      'output_3': 'sparse_categorical_accuracy'})

    checkpoint_path = "./model_checkpoints/"

    # Create checkpoint path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    train(model, X_train, y_train, X_test, y_test, checkpoint_path)


if __name__ == '__main__':
    main()
