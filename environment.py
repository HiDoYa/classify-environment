import cv2
import glob
import os
from sklearn.model_selection import train_test_split

# Loads images and returns 2 tuples for ((trainData, trainLabels), (testData, testLabels)) 
# Where data has shape (num_samples, rows, cols, depth) and rows=cols=128 for 128x128 image and depth=3 (for rgb)
# Note: Assumes image_data_format=channels_last
# Note: Returns as lists (not as numpy array)
class Environment:
    @staticmethod
    def load_data():
        tofilename = os.path.realpath(__file__)
        pathname = "/".join((tofilename.split('/')[:-1])) # Path to current directory

        # Load all images
        city_images = [cv2.imread(file) for file in glob.glob(pathname + "/images/processed/city/*.jpg")]
        forest_images = [cv2.imread(file) for file in glob.glob(pathname + "/images/processed/forest/*.jpg")]
        mountain_images = [cv2.imread(file) for file in glob.glob(pathname + "/images/processed/mountain/*.jpg")]
        ocean_images = [cv2.imread(file) for file in glob.glob(pathname + "/images/processed/ocean/*.jpg")]
        X = city_images + forest_images + mountain_images + ocean_images
        
        # Labels: 0 = city, 1 = forest, 2 = mountain, 3 = ocean
        y = [0] * len(city_images) + [1] * len(forest_images) + \
            [2] * len(mountain_images) + [3] * len(ocean_images)

        # Randomly sort and split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        return ((X_train, y_train), (X_test, y_test))

    @staticmethod
    def prediction_to_str(prediction_int):
        if prediction_int == 0:
            return "City"
        elif prediction_int == 1:
            return "Forest"
        elif prediction_int == 2:
            return "Mountain"
        elif prediction_int == 3:
            return "Ocean"

        return "Error"