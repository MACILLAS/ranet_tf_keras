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

    def build_small_net(self, inputs):
        # first downsample inputs
        x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(inputs)

        # first conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        self.small_conv_1 = x #save output for later

        # second conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        self.small_conv_2 = x # save output for later

        # classification
        x = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(x)
        x = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(x)
        x = Flatten()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=10, activation='softmax', name="output_1")(x)

        return x

    def build_med_net(self, inputs):
        # first downsample inputs
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        # first conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        temp = UpSampling2D(size=(2, 2), interpolation='bilinear')(self.small_conv_1) #upsamp lower conv
        x = Concatenate(axis=3)([x, temp]) # concat
        x = Conv2D(filters=64, kernel_size=1, padding='SAME', activation=None)(x) # reduce filters
        x = BatchNormalization()(x)
        x = ReLU()(x)
        self.med_conv_1 = x  # save output for later

        # second conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        temp = UpSampling2D(size=(2, 2), interpolation='bilinear')(self.small_conv_2)  # upsamp lower conv
        x = Concatenate(axis=3)([x, temp])  # concat
        x = Conv2D(filters=64, kernel_size=1, padding='SAME', activation=None)(x)  # reduce filters
        x = BatchNormalization()(x)
        x = ReLU()(x)
        self.med_conv_2 = x  # save output for later

        # third conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        self.med_conv_3 = x #save output for later

        # fourth conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        self.med_conv_4 = x #save output for later

        # classification
        x = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(x)
        x = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(x)
        x = Flatten()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=10, activation='softmax', name="output_2")(x)
        return x

    def build_large_net(self, inputs):
        #don't need to downsample inputs
        # first conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(inputs)
        temp = UpSampling2D(size=(2, 2), interpolation='bilinear')(self.med_conv_1)  # upsamp lower conv
        x = Concatenate(axis=3)([x, temp])  # concat
        x = Conv2D(filters=64, kernel_size=1, padding='SAME', activation=None)(x)  # reduce filters
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # second conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        temp = UpSampling2D(size=(2, 2), interpolation='bilinear')(self.med_conv_2)  # upsamp lower conv
        x = Concatenate(axis=3)([x, temp])  # concat
        x = Conv2D(filters=64, kernel_size=1, padding='SAME', activation=None)(x)  # reduce filters
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # third conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        temp = UpSampling2D(size=(2, 2), interpolation='bilinear')(self.med_conv_3)  # upsamp lower conv
        x = Concatenate(axis=3)([x, temp])  # concat
        x = Conv2D(filters=64, kernel_size=1, padding='SAME', activation=None)(x)  # reduce filters
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # fourth conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        temp = UpSampling2D(size=(2, 2), interpolation='bilinear')(self.med_conv_4)  # upsamp lower conv
        x = Concatenate(axis=3)([x, temp])  # concat
        x = Conv2D(filters=64, kernel_size=1, padding='SAME', activation=None)(x)  # reduce filters
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # fifth conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # fourth conv block
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # classification block
        x = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(x)
        x = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation=None)(x)
        x = Flatten()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=10, activation='softmax', name="output_3")(x)

        return x

    def build(self):

        inputShape = (32, 32, 3)
        # construct both the "category" and "color" sub-networks
        inputs = Input(shape=inputShape)

        smallBranch = FullRanet.build_small_net(self, inputs)
        medBranch = FullRanet.build_med_net(self, inputs)
        largeBranch = FullRanet.build_large_net(self, inputs)

        # create the model using our input (the batch of images) and
        # two separate outputs -- one for the clothing category
        # branch and another for the color branch, respectively
        model = Model(
            inputs=inputs,
            outputs=[smallBranch, medBranch, largeBranch],
            name="ranet")
        # return the constructed network architecture
        return model


model = FullRanet().build()
init_lr = 1e-3
epochs = 100
opt = Adam(lr=init_lr, decay=init_lr / epochs)

model.compile(optimizer=opt,
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
valid_batch_size = 32

checkpoint_path = "./model_checkpoints/"

# Create checkpoint path
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Keras callbacks for training
callback_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True, mode="auto", save_freq=1)
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
                    epochs=epochs, verbose=1, batch_size=batch_size, validation_batch_size=valid_batch_size, callbacks=callback_list)


















