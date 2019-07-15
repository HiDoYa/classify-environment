# Classify Environment Pictures using Lenet Neural Net Architecture
Convolutional neural network using LeNet architecture to classify a picture into 4 environment categories: forest, ocean, mountain, city. This uses Keras with tensorflow backend and is written in Python.  
Around 800 labeled sets of pictures per each category, giving a total of 3200 total sets for training and testing. Labelled sets were scraped from google images and processed with OpenCV.  
First attempt of the model gave an accuracy of 58.32% in 20 epochs.  
This model will be further improved in the future when I have access to a GPU.  

## Use:
To use, run: `python3 lenet_driver.py [options]`  
>Options:  
    -s, --save-model    Flag to train and save a model  
    -l, --load-model    Flag to load an already trained model  
    -w, --weights       Filename of the weights file for the already trained model. (Must be in output directory and hdf5 extension)  

### First attempt:
![alt text](screenshots/first_attempt.png)