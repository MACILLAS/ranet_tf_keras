"""
Med Subnetwork For Ranet
Implemented by: Max Midwinter
Mark 1.5

Based On: https://keras.io/api/models/model/
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, ReLU, Concatenate
from models.small_net import SmallModel
import os
import numpy as np

class MedModel(tf.keras.Model):

    def __init__(self):
        super(MedModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        # Load instance of small model .h5 (get_appropriate layer and those weight)
        small_model = SmallModel()
        small_model(tf.keras.Input(shape=(8, 8, 3)))
        print(os.getcwd())
        small_model.load_weights("./models/small_weights.h5")
        self.small_model = small_model

        initializer = tf.keras.initializers.Ones()

        # Define Model Layers (Yes I know this is not efficient... Fuck you.)
        # First Basic-Conv Block
        self.small_conv1 = Conv2D(filters=64, kernel_size=3, strides=1,padding='same', activation=None, name="small_conv1",kernel_initializer=self.small_conv1_init, trainable=False)
        self.small_bn1 = BatchNormalization(name="small_bn1")
        self.small_relu1 = ReLU(name="small_relu1")

        # Second Basic-Conv Block
        self.small_conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=None, name="small_conv2",kernel_initializer=self.small_conv2_init, trainable=False)
        # self.small_bn2 = BatchNormalization(name="small_bn2")
        # self.small_relu2 = ReLU(name="small_relu2")

        # Define Model Layers
        # First Conv Block
        self.med_conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', activation=None, name="med_conv1")
        self.med_concat1 = Concatenate(axis=3, name="med_concat1")
        self.med_bn1 = BatchNormalization(name="med_bn1")
        self.med_relu1 = ReLU(name="med_relu1")

        # Second Conv Block
        self.med_conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', activation=None, name="med_conv2")
        self.med_concat2 = Concatenate(axis=3, name="med_concat2")
        self.med_bn2 = BatchNormalization(name="med_bn2")
        self.med_relu2 = ReLU(name="med_relu2")

        # Third Conv Block
        self.med_conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', activation=None, name="med_conv3")
        self.med_bn3 = BatchNormalization(name="med_bn3")
        self.med_relu3 = ReLU(name="med_relu3")

        # Fourth Conv Block
        self.med_conv4 = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', activation=None, name="med_conv4")
        self.med_bn4 = BatchNormalization(name="med_bn4")
        self.med_relu4 = ReLU(name="med_relu4")

        # Classification Part
        self.med_class_conv1 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name="med_class_conv1")
        self.med_class_conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name="med_class_conv2")
        self.med_class_flatten = Flatten(name="med_class_flatten")
        self.med_class_dense = Dense(units=10, activation='softmax')

    # This function returns small_conv1_filters
    def small_conv1_init(self, shape, dtype=None):
        small_conv1_filters, biases = self.small_model.get_layer("small_conv1").get_weights()
        return small_conv1_filters

    # This function returns small_conv2_filters
    def small_conv2_init(self, shape, dtype=None):
        small_conv2_filters, biases = self.small_model.get_layer("small_conv2").get_weights()
        return small_conv2_filters

    def call(self, inputs, training=False):
        """
        The call function is inherited from Model. It defines the behaviour of the model.
        In this function we will connect the layers we defined in __init__ together.
        Please review Connection Scheme and observe naming conventions.

        :param inputs: these are the images that are passed in shape (batches, height, width, channels)
        :param training: BOOL this is a MODEL param that indicates if we are training or testing... I'm still trying to figure this out...
        :return: stuff (softmax class probabilities in this case)
        """
        ## Connect Small Network
        # Connect First Small Conv Block
        small_conv1 = self.small_conv1.apply(inputs)
        small_bn1 = self.small_bn1.apply(small_conv1)
        small_relu1 = self.small_relu1.apply(small_bn1)

        # Connect Second Small Conv Block
        small_conv2 = self.small_conv2.apply(small_relu1)

        ## Connect Med Network
        # Connect First Med Conv Block
        med_conv1 = self.med_conv1.apply(inputs)
        med_concat1 = self.med_concat1.apply([med_conv1, small_conv1])
        med_bn1 = self.med_bn1.apply(med_concat1)
        med_relu1 = self.med_relu1.apply(med_bn1)

        # Connect Second Med Conv Block
        med_conv2 = self.med_conv2.apply(med_relu1)
        med_concat2 = self.med_concat2.apply([med_conv2, small_conv2])
        med_bn2 = self.med_bn2.apply(med_concat2)
        med_relu2 = self.med_relu2.apply(med_bn2)

        # Connect Third Med Conv Block
        med_conv3 = self.med_conv3.apply(med_relu2)
        med_bn3 = self.med_bn3.apply(med_conv3)
        med_relu3 = self.med_relu3.apply(med_bn3)

        # Connect Fourth Med Conv Block
        med_conv4 = self.med_conv4.apply(med_relu3)
        med_bn4 = self.med_bn4.apply(med_conv4)
        med_relu4 = self.med_relu4.apply(med_bn4)

        # Connect Small Class Block
        med_class_conv1 = self.med_class_conv1.apply(med_relu4)
        med_class_conv2 = self.med_class_conv2.apply(med_class_conv1)
        med_class_flatten = self.med_class_flatten.apply(med_class_conv2)
        med_class_dense = self.med_class_dense.apply(med_class_flatten)

        # if training:
        #     output = med_class_dense
        # else:
        #     #pred = np.argmax(med_class_dense)
        #     #conf = np.max(med_class_dense)
        #     #output = [pred, conf]

        return med_class_dense

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
