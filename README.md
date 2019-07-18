# Classify Environment Images using Lenet Neural Net Architecture
Convolutional neural network using LeNet architecture to classify a picture into 4 environment categories: forest, ocean, mountain, city. This uses Keras with tensorflow backend and is written in Python. <br><br>
There are ~800 labeled sets of pictures per each category, giving a total of 3200 total sets for training and testing. Labelled sets were scraped from google images and processed with OpenCV.  <br><br>
I was available to achieve an accuracy of ~76% on my test set after playing with various parameters and layers. <br><br>

## Use:
To use, run: `python3 lenet_driver.py [options]`  
>Options:  
    -s, --save-model    Flag to train and save a model  
    -l, --load-model    Flag to load an already trained model  
    -w, --weights       Filename of the weights file for the already trained model. (Must be in output directory and hdf5 extension)  

## Sample Classification
![alt text](screenshots/sample1.png)<br>
city=73.03%, forest=2.20%, mountain=24.66%, ocean=0.11%
<br><br>
![alt text](screenshots/sample2.png)<br>
city=2.09%, forest=97.05%, mountain=0.80%, ocean=0.06%
<br><br>
![alt text](screenshots/sample3.png)<br>
city=0.07%, forest=1.28%, mountain=0.03%, ocean=98.62%
<br><br>
![alt text](screenshots/sample4.png)<br>
city=37.87%, forest=2.07%, mountain=48.23%, ocean=11.83%

## Analysis
My initial test accuracy was 61.47%. I noticed that the neural network seems to have trouble distinguishing city vs mountain (as reflected in the sample images above), possibly due to similar features such as tall and jagged structures. After cleaning up the dataset a little bit since some city and mountain pictures were mixed together, my test accuracy went up to 72.3%.

While my test accuracy did go up to 72.3%, my training accuracy was around 90% so I suspected that there may be some overfitting of data so I tried adding regularization via Dropout layers but found no significant improvements as my test accuracy was 72.1%. 

I then added Batch Normalization layers and removed my Dropout layers since BN and dropout are not very compatible with each other. After doing so, I found that my training accuracy rose to nearly 100% and my test accuracy went up marginally to 73.1%. There still seems to be a lot of overfitting so I played with L1, L2 regularization.

After some tinkering with the regularization rate, I was able to reach 75.96% test set accuracy using just L2 regularization (no L1).
