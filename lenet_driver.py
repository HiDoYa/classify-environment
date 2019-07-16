from lenet_model import LeNet
from environment import Environment
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import os

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1, help="(Optional) Whether to save model to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1, help="(Optional) Whether to load model from disk")
ap.add_argument("-w", "--weights", type=str, help="(Optional) Path of weights file")
args = vars(ap.parse_args())

weightsPath = "output/" + args["weights"] + ".hdf5"

# Load data (gives correct dimensions)
print("Loading training and testing sets...")
((_trainData, _trainLabels), (_testData, _testLabels)) = Environment.load_data()

# Convert to np arrays
_trainData = np.array(_trainData)
_trainLabels = np.array(_trainLabels)
_testData = np.array(_testData)
_testLabels = np.array(_testLabels)

# Scale data to [0, 1] range
trainData = _trainData.astype("float32") / 255.0
testData = _testData.astype("float32") / 255.0

# Generate vector for each label (all zeros except for one index)
# There are 4 class label
trainLabels = np_utils.to_categorical(_trainLabels, 4)
testLabels = np_utils.to_categorical(_testLabels, 4)

# Initialize optimizer and model
print("Loading model...")
opt = SGD(lr=0.01) # Stochastic gradient descent
model = LeNet.build(numChannels=3, imgRows=128, imgCols=128, numClasses=4,
                    weightsPath=weightsPath if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Trains if not loading parameters to CNN
if args["load_model"] < 0:
    print("Training...")
    model.fit(trainData, trainLabels, batch_size=32, epochs=20, verbose=1)

    print("Evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=32, verbose=1)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

# Save model (if requested)
if args["save_model"] > 0:
    print("Saving model to file...")
    if not os.path.exists("output"):
        os.mkdir("output")
    model.save_weights(weightsPath, overwrite=True)

# Randomly select a few picture to test on
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # Make prediction on picture
    # Use np.newaxis since testData[i] would give one less dimension
    probs = model.predict(testData[np.newaxis, i]) 
    prediction = probs.argmax(axis=1)
    
    # Get image from test data
    image = (testData[i] * 255).astype("uint8")
    cv2.putText(image, Environment.prediction_to_str(prediction[0]), \
            (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("Predicted: {}, Actual: {}".format(
        Environment.prediction_to_str(prediction[0]), \
        Environment.prediction_to_str(np.argmax(testLabels[i]))))
    for idx, prob in enumerate(probs[0]):
        print(Environment.prediction_to_str(idx), "=", prob*100)
    cv2.imshow("Environment", image)
    cv2.waitKey(0)

