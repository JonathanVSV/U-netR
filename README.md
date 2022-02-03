# U-netR

The repository contains several scripts used to fit a U-net model using keras (tensorflow backend) in R. 

1.Img_preprocess makes some pre preprocessing and augmentation of the data. This script transforms each pair of x (optic and radar images) and y (manually labeled images (256 x 256 pixels)) into arrays that contain the training and test set of augumented images. In this procedure for each 256 x 256 px image, 27 128 x 128 px images are generated. From these latter, 9 images correspond to cropped images (using an offset of 32 pixels), 9 to vertically mirrored images and 9 to horizontally mirrored images. Finally the images are stored in arrays and saved as an "npz" file. ImageAugument_crop4Training contains the preprocessing steps and augmentation procedures applied to the training data, prior to training the U-net with these data. CompleteImage_crop4prediction makes the necessary preprocessing steps to make a prediction over the complete study area.

2.U-net_train constructs the U-net's desired architecture, define the hyperparameters that are going to be explored, fit the model and save it. If no hyperparameter exploration wants to be made a single U-net can be constructed and trained using the U-net_train script, while the U-net_hyperparamExpl_train can be used for hyperparameters exploration.

3.ResulEval contains a script that loads the previously saved model (highest f1-score) and generates the predicted images for the test set. Additionally, the error matrix, accuracy and f1-score are calculated for the test set. In order to make the prediction over the complete study area, this same script can be used to make the predictions over the two grids generated in the CompleteImage_crop4prediction script

The 4.FinalClassification folder contains the scripts first, to make the predictions and get the probabilities of corresponding to each class for the complete study area. Then these predictions and probabilities are used to set the class as the one with the highest probability in either of the two grids used to make the predictions (Final_Unet_ensemble). Additionally, the LULC map obtained for the complete study area is available in this folder.

Preview of the final LULC classification obtained with U-net MS+SAR.

![U-net LULC](/4.FinalClassification/preview.png?raw=true "MX 1 - 3 months mosaics")
