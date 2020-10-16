library(unet)
library(keras)
library(rsample)
library(tidyverse)
library(reticulate)

# Hyperparameters for training U-net
batch_size <- 64
# Channels in features input data
channels <- 6
learn_rate <- 1e-4
epochs <- 50
dropout <- 0.5
# Number of filters in first layer of U-net
# Consecutive layers will duplicate the number of filters of the previous one
filters_firstlayer <- 64
# Number of layers for the U-net
num_layers <- 3
# Output activation function
activation_func_out <- "softmax"

#Label data; number of classes in labeled data
n_classes <- 11 

# Images dimensions in px
img_width <- 128
img_height <- 128

# Image dimensions in px for predicted ones
img_width_pred <- 128
img_height_pred <- 128

##---------------------Image Folders Definition-------------------------
#Training
#Image folder
imagery <- "OpticRadar"

#Folder containing the npz file
train_image_files_path <- file.path(paste0("Img_Preprocess"))
#---------------------------Read inputs-------------------------

# RDS option
# train_x_data <- readRDS(paste0(train_image_files_path,"/","train_x_data_OptRad.rds"))
# train_y_data <- readRDS(paste0(train_image_files_path,"/","train_y_data_OptRad.rds"))
# 
# # Aquí hay que hacer algo con los datos para que estén en un rango entre 0 y 1, quizás como un hist stretch
# test_x_data <- readRDS(paste0(train_image_files_path,"/","test_x_data_OptRad.rds"))
# test_y_data <- readRDS(paste0(train_image_files_path,"/","test_y_data_OptRad.rds"))

# Numpy zip file option
np <- import("numpy")

# Load npz
npz2 <- np$load(paste0(train_image_files_path,"/","train_test_sets"))
# See files
# npz2$files

# Load training and test data (both x: features and y: labels)
train_x_data <- npz2$f[["x_train"]]
train_y_data <- npz2$f[["y_train"]]

test_x_data <- npz2$f[["x_test"]]
test_y_data <- npz2$f[["y_test"]]

# Check some images
par(mfrow = c(2,2))
plot(train_x_data[1,,,1:3] %>% imager::as.cimg())
plot(train_x_data[1,,,4:6] %>% imager::as.cimg())
plot(train_y_data[1,,,11] %>% imager::as.cimg())
plot(train_y_data[1,,,8] %>% imager::as.cimg())


###---------------------------Model Definition---------------------------------------
# Define unet model using unet function (from unet package)
model <- unet(input_shape = c(img_width, img_height, channels),
              num_classes = n_classes,
              dropout = dropout,
              filters = filters_firstlayer,
              num_layers = num_layers,
              output_activation = activation_func_out)
# Get summary of model
summary(model)

# Define dice function to monitor this metric during training
dice <- custom_metric("dice", function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

# Compile model
model %>% compile(
  optimizer = optimizer_adam(lr = learn_rate),
  loss = "categorical_crossentropy",
  metrics = list(dice, "categorical_accuracy")
)

##-------------------------------------Run Model-------------------------------

history <- model %>% fit(
  x = train_x_data,
  y = train_y_data,
  validation_data = list(test_x_data, test_y_data),
  epochs = epochs,
  batch_size = batch_size,
  shuffle = T,
  verbose = 1)

### ---------------Save the plot containing the training phase and model-------------

# jpeg(paste0("Graficas_",epochs,"runs_",learn_rate,"lr_","model",".jpeg"),
#      #device = "jpeg",
#      width = 17,
#      height = 21,
#      units = "cm",
#      res = 100)
# plot(history,
#      theme_bw = getOption("keras.plot.history.theme_bw", FALSE))
# dev.off()

# Save model
save_model_hdf5(model, paste0("U128model",imagery,
                              "filters", filters_firstlayer,
                              "Epochs",epochs,
                              "layers",num_layers,
                              "dropout", dropout,
                              "_lr",learn_rate,
                              "_adam",".h5"))


