# U-netR

The repository contains 3 scripts to fit a U-net model using keras (tensorflow backend) in R. This repo is still under construction

Data_preprocess makes some pre preprocessing and augmentation of the data. This script transform each pair of x (optic and radar images) and y (manually labeled images (256 x 256 px)) into arrays that contain the training and test set of augumented images. In this procedure for each 256 x 256 px image, 75 128 x 128 px images are generated. From these latter, 25 images correspond to cropped images (using an offset of 32 pixels), 25 to vertically mirrored images and 25 to horizontally mirrored images. Finally the images are stored in arrays and saves as an "npz" file.

U-net_fit constructs the U-net's desired architecture, define the hyperparameters that are going to be used, fit the model and save it. 

Result_evalation loads the previously saved model and generates the predictes images for the test set. Additionally, a confusion matrix and overall accuracy is calculated for each image. Finally, some plots are generated to examine the spatial distribution of the classes and the probabilities of each pixel of belonging to the predicted class.