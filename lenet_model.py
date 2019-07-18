from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras import regularizers
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K

# Builds LeNet model
# Layers: input -> conv -> relu -> pool -> conv -> relu -> pool -> fc -> relu -> fc


class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
        # Define model and dimensions
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)
        regularizerRate = 0.0005

        # Layers
        model.add(Conv2D(20, 5, padding="same", input_shape=inputShape, kernel_regularizer=regularizers.l2(regularizerRate)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        #model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, 5, padding="same", kernel_regularizer=regularizers.l2(regularizerRate)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        #model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500, kernel_regularizer=regularizers.l2(regularizerRate)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        #model.add(Dropout(0.4))

        model.add(Dense(numClasses, kernel_regularizer=regularizers.l2(regularizerRate)))
        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
        