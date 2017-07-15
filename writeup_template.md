# Traffic Sign Recognition


## Building a Traffic Sign Recognition Pipeline

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./extra_signs/data_explore.png "Visualization"
[image2]: ./extra_signs/hist_compare.png "Histogram Equalization"
[image3]: ./extra_signs/data_normalize.png "Image Normalization"
[image4]: ./extra_signs/30_sign.jpg "Traffic Sign 1"
[image5]: ./extra_signs/bumpy_road_sign.jpg "Traffic Sign 2"
[image6]: ./extra_signs/no_passing_sign.jpg "Traffic Sign 3"
[image7]: ./extra_signs/road_work_sign.jpg "Traffic Sign 4"
[image8]: ./extra_Signs/STOP_sign.jpg "Traffic Sign 5"

---

I used vanilla python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of all three data sets (training ,validation & test) in histogram form. It is a bar chart showing how the data is distrbuted. It is noted that the distribution for all three sets is very similar to one another so data augmentation to change the data distribution is not really required to obtain a high accuracy for the validation set and test set. 

<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/data_explore.png">


# Design and Test a Model Architecture

## Data Preprocessing


Histogram equalization was conducted to increase the global and local contrast values as it is noted that there was a wide variety of images in the training set of different brightness levels. Ultimately, this allowed the ConvNet to better extract key features from the data set regardless of the brightness/contrast ratio of the image. 

Here is an example of a traffic sign image before and after histogram equalization.
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/data_hist.png">


As a last step, I normalized the image data to ensure it had equal variance and zero mean in order to better condition the optimization problem and to prevent numerical instability during training.


## ConvNet Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|		|										|
| Dropout   | keep_prob=0.5          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|		|										|
| Dropout   | keep_prob=0.5         |
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				|
| Fully connected		|  input 360x1, output 252x1        									|
| RELU					|		|										|
| Dropout  | keep_prob=0.5         |
| Fully connected		|  input 252x1, output 43x1     									|
| Softmax				|         									|
 

**Results**

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.5%
* test set accuracy of 96.6%

**The Starting Point**

The first architecture that was used was the LeNet ConvNet, which was a great starting point for this problem as it works relatively well for machine learning problems with smaller data sets like this one.(see http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) However, the LeNet architecture was designed to work designed originally to work best with small-scale OCR problems with a small number of classes (10). Hence a series of modifications had to be made to adjust the network to accurately predict and classify RGB images of traffic signs, which posed as a more complex data set for LeNet. 

**Modifications to LeNet**

The first problem with the architecture was that in order to be able to effectively  classify the RGB images to 43 distinct classes, the number of neurons or weights + biases had to increases to capture the more complex data. In order to accomplish this I simply multiplied all neurons at each layer by a factor of 3 (empirically determined), which produced slightly higher validation set accuracies. 

The next problem was that no preprocessing on the image data was implemented for the LeNet data. Hence I decided to apply histogram equalization using OpenCV and data normalization on the data sets for better feature extraction. 
  
Another issue that was noticed initially was that after these changes, the data was being overfit to the training set and the validation set accuracy seemed to be saturated at around 90%. In order to fix this issue, I first attempted to apply L2 regularization to the loss function with different values for beta along with more aggressive maxpooling operations after every activation function. However, after some experimentation, I discovered that dropout operations after every layer was more effective in preventing overfitting. The probablity of dropout applied during the training phase was set at 0.5 for each layer and during evaluation, the probablity was set to 1.0 to allow for the use of all neurons and maximize the classifying power of the neural network. Dropout forces the neural network to learn more features as there is probablity that neurons associated with any given feature maybe deactivated at any given layer.  

After applying dropout, I noticed that it took much longer to train the neural network and the initially validation accuracies would be a bit more chaotic. In order to remedy this I set a really high number for the EPOCHs and eventually settled on a value of 60 as improvements to the network after this point were very minimal or nonexistent. Occassionally, my neural network would produce validation accuracies of over 99%. This would be a sporadic occurance but regardless the network would always settle back at around 98% validation accuracy after the remaining EPOCHs. Stopping the training process at these spordic rises also produces lower test set accuracies. All in all, I believe that this occurance indicates that my model could be further optimized.  

As for the other hyper parameters, I left them practically unchanged from the original LeNet architecture (i.e. learning rate, batch size) and continued to use the AdamOptimizer for weight updates. The AdamOptimizer proved to be better than the GradientDescentOptimizer as it uses a more adaptive learning approach by factoring in an average of the previous gradients, immediate previous gradient along with the current gradient during the every weight update step. (see http://ruder.io/optimizing-gradient-descent/index.html#adam)

The two convolutional layers are very important to the networks learning ability as the lower layers have neurons that are more recipetive to local features of the data set while the higher convolutional layer uses more filters to produce higher dimensionalty data, which is useful for the final fully connected layers. A possible improvement to this network would be to change its archetecture from a strictly feed forward network to one that uses data from both the higher and lower convolutional layers for the fullly connected classifer layers. However, this would need more hyper parameter tuning to identify the size of each layer along with the number of additional layers.

All in all, I believe my network is a good choice for this classification problem as it produces accuracies of over 95% for training, validation and test sets.   

## Validating with Random Web Images 


Here are five German traffic signs that I found on the web:

<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/30_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/STOP_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/bumpy_road_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/no_passing_sign.jpg" width="200" hieght="200">
<img src="https://github.com/AbdulTheProgrammer/CarND-Traffic-Sign-Classifier-Project/blob/master/extra_signs/road_work_sign.jpg" width="200" hieght="200">


Generally these web images found using the Google search engine should not be too much of a problem for the neural network as they are similar to images in the training set. The stop sign may prove slightly challenging as the network was not trained on sheared/rotated/transformed image data. 

The following softmax probablities were output from the ConvNet: 

**Softmax Results** 

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

**Web Image Predication Accuracy** 

| Image			        |     Prediction(rounded to 1 decimal place)	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| 100.0   									| 
| 30 Sign     			| 100.0 										|
| Bumpy Road Sign					| 100.0											|
| No Passing Sign	      		| 100.0				 				|
| Roadwork Sign			| 100.0      							|

It is noted that each of my network's predication for the 5 web images is correct. My network also seems very sure of each of its predications, which is understandable due to the web image data's similarity to the training/validation set. This is evident as the largest softmax probablity for each image was >99%. This accuracy is the same as the accuracy for the training set and similar to the 98.5% accuracy for the validation set achieved during the training operation.

