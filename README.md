# U-netR

The repository contains 4 scripts to fit a U-net model using keras (tensorflow backend) in R. 

1.Img_preprocess makes some pre preprocessing and augmentation of the data. This script transform each pair of x (optic and radar images) and y (manually labeled images (256 x 256 px)) into arrays that contain the training and test set of augumented images. In this procedure for each 256 x 256 px image, 75 128 x 128 px images are generated. From these latter, 25 images correspond to cropped images (using an offset of 32 pixels), 25 to vertically mirrored images and 25 to horizontally mirrored images. Finally the images are stored in arrays and saves as an "npz" file. ImageAugument_crop4Training contains the preprocessing steps and augmentation procedures applied to the training data, prior to training the U-net with these data. CompleteImage_crop4prediction makes the necessary preprocessing steps to make a prediction over the complete study area.

2.U-net_train constructs the U-net's desired architecture, define the hyperparameters that are going to be explored, fit the model and save it. If no hyperparameter exploration wants to be made a single U-net can be constructed and trained using the U-net_train script, while the U-net_hyperparamExpl_train can be used for hyperparameters exploration.

3.ResulEval contains a script that loads the previously saved model (highest f1-score) and generates the predicted images for the test set. Additionally, the error matrix, accuracy and f1-score are calculated for the test set.

The 4.FinalClassification folder contains the LULC obtained for the complete study area.