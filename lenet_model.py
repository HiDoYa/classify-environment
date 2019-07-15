from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# Builds LeNet model
# Layers: input -> conv -> relu -> pool -> conv -> relu -> pool -> fc -> relu -> fc


class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
        # Define model and dimensions
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # Layers
        model.add(Conv2D(20, 5, padding="same", input_shape=inputShape)) # 20 convolution filters of size 5x5
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, 5, padding="same")) # 50 convolution filters of size 5x5
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
        