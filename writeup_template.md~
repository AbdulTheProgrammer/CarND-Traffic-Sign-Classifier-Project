#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_explore.png "Visualization"
[image2]: ./examples/hist_compare.png "Histogram Equalization"
[image3]: ./extra_signs/30_sign.jpg "Traffic Sign 1"
[image4]: ./extra_signs/bumpy_road_sign.jpg "Traffic Sign 2"
[image5]: ./extra_signs/no_passing_sign.jpg "Traffic Sign 3"
[image6]: ./extra_signs/road_work_sign.jpg "Traffic Sign 4"
[image7]: ./extra_Signs/STOP_sign.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

I used vanilla python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of all three data sets (training ,validation & test) in histogram form. It is a bar chart showing how the data. It is noted that the distribution for all three sets is very similar so data augmentation to change the data distribution is not really required. 

[image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to use histogram equalization on all three data sets along with local adaptive histogram equalization using a tile size of (8,8) (this tile value was reached using an empirical / experimental process to achieve the best accuracy). This histogram equalization step was applied to the H channel of each HSV converted image (usually histogram equalization is conducted on grayscale images). Histogram equalization was conducted to increase the global and local contrast values as it was noted that there were a wide variety of images in the training set of different brightness levels. Ultimately, this allowed the ConvNet to better extract key features from the data set. 

Here is an example of a traffic sign image before and after histogram equalization.

[image2]

As a last step, I normalized the image data to ensure it had equal variance and zero mean in order to better condition the optimization problem and to prevent numerical instability during training. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Dropout 0.5  |           |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|												|
| Dropout  0.5 |          |
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				|
| Fully connected		|  input 360x1, output 252x1        									|
| RELU					|												|
| Dropout  0.5 |          |
| Fully connected		|  input 252x1, output 43x1        									|
| Softmax				|         									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a modified LeNet Archtecture, which I developed after some reaserach and empirical testing. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.5%
* test set accuracy of 96.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/30_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/STOP_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/bumpy_road_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/no_passing_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/road_work_sign.jpg" width="200" hieght="200">


Generally these web images found using the Google search engine should not be too much of a problem for the neural network as they are similar to images in the training set. The stop sign may prove slightly challenging as the network was not trained on sheared/rotated/transformed image data. 

The following softmax probablities were output from the ConvNet: 

TopKV2(values=array([[  9.99821126e-01,   1.78703878e-04,   1.45523117e-07,
          7.37584926e-08,   5.54541124e-08],
       [  1.00000000e+00,   4.25815321e-11,   8.36842401e-12,
          1.49889457e-15,   1.28643267e-15],
       [  9.99999404e-01,   5.98163581e-07,   5.14689347e-10,
          2.08560252e-10,   1.39160883e-12],
       [  1.00000000e+00,   9.04436411e-13,   1.47629128e-13,
          4.47968222e-18,   2.04703979e-18],
       [  1.00000000e+00,   2.01609329e-09,   2.24924720e-13,
          1.32523461e-15,   1.79794879e-16]], dtype=float32), indices=array([[14, 13, 17,  1, 42],
       [ 9, 19, 10, 20, 41],
       [25, 31, 29, 30, 21],
       [22, 17,  0, 25, 31],
       [ 1, 14,  5, 35,  0]], dtype=int32))

| Image			        |     Prediction(rounded to 1 decimal place)	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| 100.0   									| 
| 30 Sign     			| 100.0 										|
| Bumpy Road Sign					| 100.0											|
| No Passing Sign	      		| 100.0				 				|
| Roadwork Sign			| 100.0      							|

It is noted that the network seems very sure of each of its predications, which is understandable due to the image data's similarity to the training/validation set as the largest softmax probablity for each image was >99%, which is the same at the accuracy for the train set and similar to the 98.5% accuracy for the validation set.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


