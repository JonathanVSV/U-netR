# U-netR

The repository contains several scripts used to fit a U-net model using keras (tensorflow backend) in R. The article where these analyses are presented can be consulted in: Solórzano, J. V., Mas, J. F., Gao, Y., Gallardo-Cruz, J. A. (2021). Land Use Land Cover Classification with U-Net: Advantages of Combining Sentinel-1 and Sentinel-2 Imagery. Remote Sensing, 13, 3600. https://doi.org/10.3390/rs13183600

The structure of the repository is the following:

1.Img_preprocess makes some pre preprocessing and augmentation of the data. This script transforms each pair of x (optic and radar images) and y (manually labeled images (256 x 256 pixels)) into arrays that contain the training and test set of augumented images. In this procedure for each 256 x 256 px image, 27 128 x 128 px images are generated. From these latter, 9 images correspond to cropped images (using an offset of 32 pixels), 9 to vertically mirrored images and 9 to horizontally mirrored images. Finally the images are stored in arrays and saved as an "npz" file. ImageAugument_crop4Training contains the preprocessing steps and augmentation procedures applied to the training data, prior to training the U-net with these data. CompleteImage_crop4prediction makes the necessary preprocessing steps to make a prediction over the complete study area.

2.U-net_train constructs the U-net's desired architecture, define the hyperparameters that are going to be explored, fit the model and save it. If no hyperparameter exploration wants to be made a single U-net can be constructed and trained using the U-net_train script, while the U-net_hyperparamExpl_train can be used for hyperparameters exploration.

3.ResulEval contains a script that loads the previously saved model (highest avgf1-score) and generates the predicted images for the test set. Additionally, the error matrix, accuracy and avgf1-score are calculated for the test set. In order to make the prediction over the complete study area, this same script can be used to make the predictions over the two grids generated in the CompleteImage_crop4prediction script

The 4.FinalClassification folder contains two scripts. The first one (Predict_Fullimage.R), makes the class predictions and get the probabilities of corresponding to each class for the complete study area. This script makes the predictions using two different grids, where the center of one grid overlaps with the edges of the other. Then these predictions and probabilities are used to set the class as the one with the highest probability in either of the two grids used to make the predictions (Final_Unet_ensemble). Additionally, the LULC map obtained for the complete study area is available in this folder.

Visualization of the U-Net model, made with [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

![U-net LULC](/4.FinalClassification/unet2d.jpg?raw=true "U-net diagram")

Preview of the final LULC classification obtained with U-net MS+SAR.

![U-net LULC](/4.FinalClassification/preview.png?raw=true "LULC classification")
