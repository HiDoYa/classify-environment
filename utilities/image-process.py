import cv2
import numpy
import sys
import os

os.mkdir('images')
errfile  = open("err.txt", "w")

# Passed in all images as arg
img_names = sys.argv[1:]

for idx, img_name in enumerate(img_names):
    try:
        image = cv2.imread(img_name)
        image = cv2.resize(image, (128, 128))
        cv2.imwrite("images/" + str(idx) + ".jpg", image)
    except:
        errfile.write("Error for file:" + img_name + '\n')

errfile.close()