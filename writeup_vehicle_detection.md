## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

In this implementation, it includes for parts:

- collecting training images, vehicles and non-vehicles
- selecting combined features, including spatial feature, color feature and hog feature
- selecting linear svm model
- spliting training images into training set and test set
- starting training
- dumping result of training
- checking example images
- checking project video 


---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

my full code is in the file "vehicle_classify_v1.ipynb"

code of extract hog features is in the function "get_hog_features"

The code for this step is contained in the second code cell of the IPython notebook file "vehicle_classify_v1.ipynb"

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

example of vehicle image:

![vehicle image](train_data/vehicles/GTI_Far/image0039.png)

example of non-vehicle image:

![vehicle image](train_data/non-vehicles/GTI/image1024.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![hog examples](output_images/hog_example.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

i checked lots of images for training, the size of those images is 64*64. so, i select hog paramter pixels_per_cell = 8, cells_per_block = 2, orientations = 9

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using those images under directory "train_data"

non-vehicle images is the same directory and vehicle images is in the another directory

i combined spatial feature, color feature and hog feature.

code of extract spatial feature is in the function "get_spatial_features"

code of extract color feature is in the function "get_color_features"

code of extract hog feature is in the function "get_hot_features"

extract features takes 73 seconds

time of SVC training: 17 seconds

accuracy of SVC classify:  0.982

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

code of sliding window search is in the function "find_cars"

i test and check example images by search multi-scale window, i select
three scale: 1, 1.5, 2

my test code is in the function "test_example_images_for_multi_scales"

![search scale 1.0](output_images/search_window_scale10.png)
![search scale 1.5](output_images/search_window_scale15.png)
![search scale 2.0](output_images/search_window_scale20.png)


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

examples of only one scale:

![search test1](output_images/sliding_box_test1.png)

![search test2](output_images/sliding_box_test2.png)

![search test3](output_images/sliding_box_test3.png)

![search test4](output_images/sliding_box_test4.png)

![search test5](output_images/sliding_box_test5.png)

![search test6](output_images/sliding_box_test6.png)

---

examples of mixed multi scales:

![search with multi scales](output_images/search_window_multi_scale.png)

examples of add svc threshold (0.7):

![search with svc threshold](output_images/search_window_multi_scale_svc_threshold.png)


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my test video result](./output_videos/test_video.mp4) [link to my project video result](./output_videos/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

video frame images is saved in the directory "./test_video_images"

code of test video frame images is in the function "test_video_images"

### Here are six frames and their corresponding heatmaps:

![video frame test1](output_images/video_frame_test1.png)
![video frame test2](output_images/video_frame_test2.png)
![video frame test3](output_images/video_frame_test3.png)
![video frame test4](output_images/video_frame_test4.png)
![video frame test5](output_images/video_frame_test5.png)
![video frame test6](output_images/video_frame_test6.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

- yolo framework is more better than svm and hog feature, it could get correctness above 99.6%, in the future, i would like to try it
- in this project, outline box (y_start, y_stop) of search window is fixed, it was not so good

