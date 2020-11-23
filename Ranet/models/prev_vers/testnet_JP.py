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
import os
from tensorflow.keras import layers
import tensorflow as tf

class FullRanet(tf.keras.Model):

    def __init__(self):
        super(FullRanet, self).__init__()
        'Shared Filters List'
        self.small_conv_1 = None
        self.small_conv_2 = None
        self.med_conv_1 = None
        self.med_conv_2 = None
        self.med_conv_3 = None
        self.med_conv_4 = None

        # first downsample inputs
        self.block_0 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))

        # first conv block
        block_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        block_1 = BatchNormalization()(block_1)
        block_1 = ReLU()(block_1)
        self.block_1 = block_1  # save output for later

        # second conv block
        block_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        block_2 = BatchNormalization()(block_2)
        block_2 = ReLU()(block_2)
        self.block_2 = block_2  # save output for later

        # classification
        block_3 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)
        block_3 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(block_3)
        block_3 = Flatten()(block_3)
        self.block_3 = Dense(units=10, activation='softmax', name="output_1")(block_3)

        # create the model using our input (the batch of images) and
        # two separate outputs -- one for the clothing category
        # branch and another for the color branch, respectively

    def call(self, inputs):
        # your own forward
        x = self.block_0(inputs)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x


model = FullRanet()
model(tf.keras.Input(shape=(32, 32, 3)))

model.summary()
init_lr = 1e-3
epochs = 100
opt = Adam(lr=init_lr, decay=init_lr / epochs)
model.compile(optimizer=opt, loss={'output_1': 'sparse_categorical_crossentropy'}, metrics={'output_1': 'sparse_categorical_accuracy'})


# model.compile(optimizer=opt,
#               loss={
#                   'output_1': 'sparse_categorical_crossentropy',
#                   'output_2': 'sparse_categorical_crossentropy',
#                   'output_3': 'sparse_categorical_crossentropy'},
#               loss_weights={
#                   'output_1': 1,
#                   'output_2': 1,
#                   'output_3': 1},
#               metrics={
#                   'output_1': 'sparse_categorical_accuracy',
#                   'output_2': 'sparse_categorical_accuracy',
#                   'output_3': 'sparse_categorical_accuracy'})
#
# batch_size = 64
# valid_batch_size = 64
#
# checkpoint_path = "./model_checkpoints/"
#
# # Create checkpoint path
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)
#
# # Keras callbacks for training
# callback_list = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_path + \
#                 "weights.e{epoch:02d}-" + \
#                 "acc{val_sparse_categorical_accuracy:.4f}.h5",
#         monitor='val_output_3_sparse_categorical_accuracy',
#         save_best_only=True,
#         save_weights_only=True, mode="auto", period=1)
# ]
#
# # Setting class names for the dataset
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# # Loading the dataset
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#
# # normalize images
# X_train = X_train / 255.0
# X_test = X_test / 255.0
#
#
# history = model.fit(x=X_train, y={"output_1": y_train, "output_2": y_train, "output_3": y_train},
#                     validation_data=(X_test, {"output_1": y_test, "output_2": y_test, "output_3": y_test}),
#                     epochs=epochs, verbose=1, batch_size=batch_size, validation_batch_size=valid_batch_size)


















