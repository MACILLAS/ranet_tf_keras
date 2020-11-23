"""
Small Subnetwork For Ranet
Implemented by: Max Midwinter
Mark 1.5

Based On: https://keras.io/api/models/model/
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, ReLU
import numpy as np

class SmallModel(tf.keras.Model):

    def __init__(self):
        super(SmallModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        # Define Model Layers (Yes I know this is not efficient... Fuck you.)
        # First Basic-Conv Block
        self.small_conv1 = Conv2D(filters=64, kernel_size=3, strides=1,padding='same', activation=None, name="small_conv1")
        self.small_bn1 = BatchNormalization(name="small_bn1")
        self.small_relu1 = ReLU(name="small_relu1")

        # Second Basic-Conv Block
        self.small_conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=None, name="small_conv2")
        self.small_bn2 = BatchNormalization(name="small_bn2")
        self.small_relu2 = ReLU(name="small_relu2")

        # Classification Part
        self.small_class_conv1 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name="small_class_conv1")
        self.small_class_conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name="small_class_conv2")
        self.small_class_flatten = Flatten(name="small_class_flatten")
        self.small_class_dense = Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):
        """
        The call function is inherited from Model. It defines the behaviour of the model.
        In this function we will connect the layers we defined in __init__ together.
        Please review Connection Scheme and observe naming conventions.

        :param inputs: these are the images that are passed in shape (batches, height, width, channels)
        :param training: BOOL this is a MODEL param that indicates if we are training or testing... I'm still trying to figure this out...
        :return: stuff (softmax class probabilities in this case)
        """

        # Connect First Small Conv Block
        small_conv1 = self.small_conv1.apply(inputs)
        small_bn1 = self.small_bn1.apply(small_conv1)
        small_relu1 = self.small_relu1.apply(small_bn1)

        # Connect Second Small Conv Block
        small_conv2 = self.small_conv2.apply(small_relu1)
        small_bn2 = self.small_bn2.apply(small_conv2)
        small_relu2 = self.small_relu2.apply(small_bn2)

        # Connect Small Class Block
        small_class_conv1 = self.small_class_conv1.apply(small_relu2)
        small_class_conv2 = self.small_class_conv2.apply(small_class_conv1)
        small_class_flatten = self.small_class_flatten.apply(small_class_conv2)
        small_class_dense = self.small_class_dense.apply(small_class_flatten)

        # if training:
        #     output = small_class_dense
        # else:
        #     #pred = np.argmax(small_class_dense)
        #     #conf = np.max(small_class_dense)
        #     #output = [pred, conf]

        return small_class_dense

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
