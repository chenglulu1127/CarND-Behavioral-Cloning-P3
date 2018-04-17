# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/model_vis.png "Model Visualization"
[image2]: ./plots/model_mse.png "Model Performance"
[image3]: ./plots/center_2018_04_16_13_29_12_036.jpg "Centerd Image"
[image4]: ./plots/right_2018_04_16_13_29_12_036.jpg "Right Image"
[image5]: ./plots/left_2018_04_16_13_29_12_036.jpg "Left Image"
[image6]: ./plots/flipped_img.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of:
 - 2 convolution with 5x5 filter sizes and depths 24 and 36 (model.py lines 77-78) 
 - 2 convolution with 3x3 filter sizes and depths 48 and 64 (model.py lines 79-80) 
 - Flatten (model.py line 81)
 - 4 Fully connected layer (model.py lines 82-85)

The model includes RELU layers to introduce nonlinearity (code line 77-80), and the data is normalized in the model using a Keras lambda layer (code line 75). 

#### 2. Attempts to reduce overfitting in the model

I use fewer epochs to reduce overfitting. I tried dropout layer but it doesn't improve the performance. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 88-89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, side images and flipped images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model that has 4 conv layers with relu activation and 4 fully connected layers. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on both the training set and validation set. 

However when I imported the model to the simulator to run the autonomous mode, it always turned very little angle and fell off the road. I knew it must be the data collection problem. I'll further explain in section 3. 

After I collected better dataset, I ran 20 epochs to train. But validation set accuracy started to go up after 16 epochs. To avoid overfitting, I terminated the training at 15 epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-85) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x4 image   							| 
| Lambda         		| lambda x: x/255.0 - 0.5   							| 
| Cropping         		| Crop out top 40 an bottom 20  							| 
| Convolution 5x5     	| 2x2 stride, depth 24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, depth 36 	|
| RELU					|												|
| Convolution 3x3	    | 2x2 stride, depth 48 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, depth 64 	|
| RELU					|												|
| Fully connected		| outputs 100        									|
| Fully connected		| outputs 50        									|
| Fully connected		| outputs 10        									|
| Softmax				|         									|
|						|												|
|						|												|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The creation of the training set was the hardest process for me.... I was a really bad player of car racing games.... and always drove outside the road and ended up with hitting something.... In order to collect the dataset, I practiced the whole evening... Finally I could keep my car staying on the road..... 

However, the way I did it was driving slowly and continuously pressing "<-" button to turn. But this actually caused a big problem for training the model because the steering angle would be always around 0.1-0.5... That's why the first model I trained had low training and validation accuracy but always fell off the road in autonomous mode. It coudn't succesffuly turn on big turns.  

So I collected data again and this time I practiced and could really drive like a driver. The steering angles looks good. And I used this training dataset to train a well behaved autonomous car.

The final training dataset I recorded is three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

To augment the dataset, I flipped images and angles thinking that this would help generalize the model and reduce the bias of more left turns. For example, here is an image that has then been flipped:

![alt text][image6]

I also augment the dataset by using the left and right side images and adding/minusing a correction of 0.5 to the center angles. Here are images of left and right side images of the centered image shown above:

![alt text][image5]
![alt text][image4]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Model Performance
Here's a visualization of train and validation set performance:

![alt text][image2]

For live test performance on simulator, please refer to run.mp4.
