"""
FullRanet
Implemented by: Max Midwinter
Mark 9.1

Latest: 2020-11-23
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import numpy as np

import tensorflow as tf


class FullRanet(tf.keras.Model):

    def __init__(self):
        super(FullRanet, self).__init__()

        # Define Classification Threshold
        self.threshold = 0.7

        # Define the layers here
        self.small_input = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
        self.med_input = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

        # The first conv2d layer in (small, med and large) models
        self.conv_in1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.conv_in2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.conv_in3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        # Remaining conv2d layers
        self.small_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.med_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.med_conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.med_conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)

        # Batch normalization and relu layers
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()
        self.bn4 = BatchNormalization()
        self.relu4 = ReLU()
        self.bn5 = BatchNormalization()
        self.relu5 = ReLU()
        self.bn6 = BatchNormalization()
        self.relu6 = ReLU()
        self.bn7 = BatchNormalization()
        self.relu7 = ReLU()
        self.bn8 = BatchNormalization()
        self.relu8 = ReLU()
        self.bn9 = BatchNormalization()
        self.relu9 = ReLU()
        self.bn10 = BatchNormalization()
        self.relu10 = ReLU()
        self.bn11 = BatchNormalization()
        self.relu11 = ReLU()
        self.bn12 = BatchNormalization()
        self.relu12 = ReLU()

        # Classification Conv2d layers
        self.class_small_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_small_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_med_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_med_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_large_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_large_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')

        # reduction layers
        self.conv_red1 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red2 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red3 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red4 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red5 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red6 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)

        # Flatten, dropout and dense layers
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)
        self.dropout4 = Dropout(0.2)
        self.dropout5 = Dropout(0.2)
        self.dropout6 = Dropout(0.2)
        self.dense_1 = Dense(units=10, activation='softmax', name="output_1")
        self.dense_2 = Dense(units=10, activation='softmax', name="output_2")
        self.dense_3 = Dense(units=10, activation='softmax', name="output_3")

        # Merge Layers (upsample2D and concatenate)
        self.upsamp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat1 = Concatenate(axis=-1)
        self.upsamp2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat2 = Concatenate(axis=-1)
        self.upsamp3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat3 = Concatenate(axis=-1)
        self.upsamp4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat4 = Concatenate(axis=-1)
        self.upsamp5 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat5 = Concatenate(axis=-1)
        self.upsamp6 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat6 = Concatenate(axis=-1)

    def build(self, input_shapes):
        inputs = Input(shape=(32, 32, 3))  # for CIFAR10 datas

        self.output_1, self.small_1, self.small_2 = self.build_small_net(inputs=inputs)
        self.output_2, self.med_1, self.med_2, self.med_3, self.med_4 = self.build_med_net(inputs=inputs,
                                                                                           small_conv_1=self.small_1,
                                                                                           small_conv_2=self.small_2)
        self.output_3 = self.build_large_net(inputs=inputs, med_conv_1=self.med_1, med_conv_2=self.med_2,
                                             med_conv_3=self.med_3, med_conv_4=self.med_4)
        return Model(inputs, [self.output_1, self.output_2, self.output_3])

    def call(self, inputs, training=None):

        self.output_1, self.small_1, self.small_2 = self.build_small_net(inputs=inputs)
        self.output_2, self.med_1, self.med_2, self.med_3, self.med_4 = self.build_med_net(inputs=inputs,
                                                                                           small_conv_1=self.small_1,
                                                                                           small_conv_2=self.small_2)
        self.output_3 = self.build_large_net(inputs=inputs, med_conv_1=self.med_1, med_conv_2=self.med_2,
                                             med_conv_3=self.med_3, med_conv_4=self.med_4)
        return [self.output_1, self.output_2, self.output_3]

    def build_small_net(self, inputs):
        # first downsample inputs
        x = self.small_input(inputs)

        # first conv block
        x = self.conv_in1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        small_conv_1 = x  # save output for later

        # second conv block
        x = self.small_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        small_conv_2 = x  # save output for later

        # classification
        x = self.class_small_conv1(x)
        x = self.class_small_conv2(x)
        x = self.flatten1(x)
        x = self.dropout1(x)
        x = self.dense_1(x)

        return [x, small_conv_1, small_conv_2]

    def build_med_net(self, inputs, small_conv_1, small_conv_2):
        # first downsample inputs
        x = self.med_input(inputs)
        # first conv block
        x = self.conv_in2(x)
        temp = self.upsamp1(small_conv_1)
        x = self.concat1([x, temp])
        x = self.conv_red1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        med_conv_1 = x  # save output for later

        # second conv block
        x = self.med_conv2(x)
        temp = self.upsamp2(small_conv_2)
        x = self.concat2([x, temp])
        x = self.conv_red2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        med_conv_2 = x  # save output for later

        # third conv block
        x = self.med_conv3(x)
        x = self.bn5(x)
        x = self.relu5(x)
        med_conv_3 = x  # save output for later

        # fourth conv block
        x = self.med_conv4(x)
        x = self.bn6(x)
        x = self.relu6(x)
        med_conv_4 = x  # save output for later

        # classification
        x = self.class_med_conv1(x)
        x = self.class_med_conv2(x)
        x = self.flatten2(x)
        x = self.dropout2(x)
        x = self.dense_2(x)
        return [x, med_conv_1, med_conv_2, med_conv_3, med_conv_4]

    def build_large_net(self, inputs, med_conv_1, med_conv_2, med_conv_3, med_conv_4):
        # don't need to downsample inputs
        # first conv block
        x = self.conv_in3(inputs)
        temp = self.upsamp3(med_conv_1)
        x = self.concat3([x, temp])
        x = self.conv_red3(x)
        x = self.bn7(x)
        x = self.relu7(x)

        # second conv block
        x = self.large_conv2(x)
        temp = self.upsamp4(med_conv_2)
        x = self.concat4([x, temp])
        x = self.conv_red4(x)
        x = self.bn8(x)
        x = self.relu8(x)

        # third conv block
        x = self.large_conv3(x)
        temp = self.upsamp5(med_conv_3)
        x = self.concat5([x, temp])
        x = self.conv_red5(x)
        x = self.bn9(x)
        x = self.relu9(x)

        # fourth conv block
        x = self.large_conv4(x)
        temp = self.upsamp6(med_conv_4)
        x = self.concat6([x, temp])
        x = self.conv_red6(x)
        x = self.bn10(x)
        x = self.relu10(x)

        # fifth conv block
        x = self.large_conv5(x)
        x = self.bn11(x)
        x = self.relu11(x)

        # sixth conv block
        x = self.large_conv6(x)
        x = self.bn12(x)
        x = self.relu12(x)

        # classification block
        x = self.class_large_conv1(x)
        x = self.class_large_conv2(x)
        x = self.flatten3(x)
        x = self.dropout3(x)
        x = self.dense_3(x)
        return x

    def predict(self, x):
        # Run small_net
        output_1, small_1, small_2 = self.build_small_net(x)
        out_vector = output_1.numpy()
        pred = np.argmax(out_vector)
        conf = out_vector[0, pred]

        # If conf of small_net less than self.threshold... Run med_net
        if conf < self.threshold:
            output_2, med_1, med_2, med_3, med_4 = self.build_med_net(x, small_conv_1=small_1, small_conv_2=small_2)
            out_vector = output_2.numpy()
            pred = np.argmax(out_vector)
            conf = out_vector[0, pred]
            # If conf of med_net is less than self.threshold... Run large_net
            if conf < self.threshold:
                output_3 = self.build_large_net(x, med_conv_1=med_1, med_conv_2=med_2, med_conv_3=med_3,
                                                med_conv_4=med_4)
                out_vector = output_3.numpy()
                pred = np.argmax(out_vector)
                conf = out_vector[0, pred]

        return [pred, conf]

"""
This is how we run everything
"""
model = FullRanet()

init_lr = 1e-4
epochs = 10

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

batch_size = 64
valid_batch_size = 64

checkpoint_path = "./model_checkpoints/"

# Create checkpoint path
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    elif epoch < 250:
        return lr * 0.1
    else:
        return lr * 0.01


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

# Setting class names for the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

history = model.fit(x=X_train, y={"output_1": y_train, "output_2": y_train, "output_3": y_train},
                    validation_data=(X_test, {"output_1": y_test, "output_2": y_test, "output_3": y_test}),
                    epochs=epochs, verbose=1, batch_size=batch_size, validation_batch_size=valid_batch_size,
                    callbacks=callback_list)

pred, conf = model.predict(x=X_test[0].reshape(1, 32, 32, 3))
y = y_test[0]
