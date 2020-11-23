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

        # downsample inputs
        self.small_input = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
        self.med_input = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

        # conv layers
        self.conv_in1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.conv_in2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.conv_in3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        #self.conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.small_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.med_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.med_conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.med_conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)
        self.large_conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)

        # reduction layers
        self.conv_red1 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red2 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red3 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red4 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red5 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)
        self.conv_red6 = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)

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

        # class layers
        self.class_small_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_small_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_med_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_med_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_large_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')
        self.class_large_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', strides=2, activation='relu')

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

        # merge layers
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

    def call(self, inputs, training=False):
        # Small Network
        # Conv Block 1
        x = self.small_input(inputs)
        x = self.conv_in1(x)
        x = self.bn11(x) #bn1
        x = self.relu1(x) #relu1
        self.small_conv_1 = x
        # Conv Block 2
        x = self.small_conv2(x) #small_conv2
        x = self.bn2(x) #bn2
        x = self.relu2(x) #relu2
        self.small_conv_2 = x
        # Classification Block
        x = self.class_small_conv1(x) #class_small_conv1
        x = self.class_small_conv2(x) #class_small_conv2
        x = self.flatten1(x) #flatten1
        x = self.dropout1(x) #dropout1
        x = self.dense_1(x)

        #Medium Network
        # Conv 1
        y = self.med_input(inputs)
        y = self.conv_in2(y)
        y = self.bn3(y) #bn3
        y = self.relu3(y) #relu3
        temp1 = self.upsamp1(self.small_conv_1) #upsamp1
        y = self.concat1([y, temp1]) #concat1
        #x = self.conv_red1(x) # conv_red1
        self.med_conv_1 = y
        # Conv 2
        y = self.med_conv2(y) # med_conv2
        y = self.bn4(y) #bn4
        y = self.relu4(y) #relu4
        temp2 = self.upsamp2(self.small_conv_2) #upsamp2
        y = self.concat2([y, temp2]) #concat2
        #x = self.conv_red2(x) # conv_red2
        self.med_conv_2 = y
        # Conv 3
        y = self.med_conv3(y) #med_conv3
        y = self.bn5(y) #bn5
        y = self.relu5(y) #relu5
        self.med_conv_3 = y
        # Conv 4
        y = self.med_conv4(y) # med_conv4
        y = self.bn6(y) #bn6
        y = self.relu6(y) #relu6
        self.med_conv_4 = y
        # Classification Block
        y = self.class_med_conv1(y) # class_med_conv1
        y = self.class_med_conv2(y) # class_med_conv2
        y = self.flatten2(y) #flatten2
        y = self.dropout2(y) #dropout2
        y = self.dense_2(y)

        #Large Model
        # Conv 1
        z = self.conv_in3(inputs)
        z = self.bn7(z) #bn7
        z = self.relu7(z) #relu7
        temp3 = self.upsamp3(self.med_conv_1) #upsamp3
        z = self.concat3([z, temp3]) #concat3
        #x = self.conv_red3(x) #conv_red3
        # Conv 2
        z = self.large_conv2(z) #large_conv2
        z = self.bn8(z) #bn8
        z = self.relu8(z) #relu8
        temp4 = self.upsamp4(self.med_conv_2) #upsamp4
        z = self.concat4([z, temp4]) #concat4
        #x = self.conv_red4(x) #conv_red4
        # Conv 3
        z = self.large_conv3(z) #large_conv3
        z = self.bn9(z) #bn9
        z = self.relu9(z) #relu9
        temp5 = self.upsamp5(self.med_conv_3) #upsamp5
        z = self.concat5([z, temp5]) #concat5
        #x = self.conv_red5(x) #conv_red5
        # Conv 4
        z = self.large_conv4(z) #large_conv4
        z = self.bn10(z) #bn10
        z = self.relu10(z) #relu10
        temp6 = self.upsamp6(self.med_conv_4) #upsamp6
        z = self.concat6([z, temp6]) #concat6
        #x = self.conv_red6(x) #conv_red6
        # Conv 5
        z = self.large_conv5(z) #large_conv5
        z = self.bn11(z) #bn11
        z = self.relu11(z) #relu11
        # Conv 6
        z = self.large_conv6(z) #large_conv6
        z = self.bn12(z) #bn12
        z = self.relu12(z) #relu12
        # Classification Block
        z = self.class_large_conv1(z) #class_large_conv1
        z = self.class_large_conv2(z) #class_large_conv2
        z = self.flatten3(z) #flatten3
        z = self.dropout3(z) #dropout3
        z = self.dense_3(z)

        return [x, y, z]

model = FullRanet()
init_lr = 1e-4
epochs = 100
opt = Adam(lr=init_lr, decay=init_lr / epochs)

model.compile(optimizer=opt,
              loss={
                  'output_1': 'sparse_categorical_crossentropy',
                  'output_2': 'sparse_categorical_crossentropy',
                  'output_3': 'sparse_categorical_crossentropy'},
              loss_weights={
                  'output_1': 0.3,
                  'output_2': 0.3,
                  'output_3': 0.3},
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

# Keras callbacks for training
callback_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True, mode="auto", save_freq=50)
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


















