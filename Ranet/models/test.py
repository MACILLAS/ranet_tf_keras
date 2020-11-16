# Convolutional Neural Network
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
visible = Input(shape=(64, 64, 1))
something = Conv2D(32, kernel_size=4, activation='relu')#(visible)
conv1 = something.apply(visible)
pool1 = MaxPooling2D(pool_size=(2, 2)).apply(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu').apply(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2)).apply(conv2)
flat = Flatten().apply(pool2)
hidden1 = Dense(10, activation='relu').apply(flat)
output = Dense(1, activation='sigmoid').apply(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='convolutional_neural_network.png')