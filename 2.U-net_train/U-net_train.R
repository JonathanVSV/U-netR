# Load libraries

library(unet)
library(keras)
library(reticulate)
library(tfruns)

np <- import("numpy")

##---------------------Constant def---------------------------------------

#Training data
# batch_size <- 64
# This might need to be adjusted according to the type of imgery used to train 
# the unet
channels <- 6
# learn_rate <- 1e-4
# epochs <- 40
# dropout <- 0.2
# filters_firstlayer <- 64
# num_layers <- 3
activation_func_out <- "softmax"

# Hyperparameter flags 
# This values are going to be filled when they are called by the tfruns script,
# i.e., U-net_hyperparamExpl_train script
FLAGS <- flags(
  flag_numeric("batch_size", 0),
  flag_numeric("learn_rate", 0),
  flag_numeric("epochs", 0),
  flag_numeric("dropout", 0),
  flag_numeric("filters_firstlayer",0),
  flag_numeric("num_layers", 0)
)


# Label data
n_classes <- 12 

# dimensions of the images that are going to be used by the unet
img_width <- 128
img_height <- 128

# dimensions of the images that are being predicted by the unet
img_width_pred <- 128
img_height_pred <- 128

##---------------------Image Folders Definition-------------------------
#Training

#Image folder
imagery <- "OpticRadar"

#Folder de training 
train_image_files_path <- file.path(paste0("training"))

#---------------------------Check image read-------------------------
# Load npz
npz2 <- np$load(paste0(train_image_files_path,"/","LULC_",input_type,"_Augment9perim_samples",".npz"))
# See files
#npz2$files

# Load training data
train_x_data <- npz2$f[["x_train"]]
train_y_data <- npz2$f[["y_train"]]

# Load test data
test_x_data <- npz2$f[["x_test"]]
test_y_data <- npz2$f[["y_test"]]

# par(mfrow = c(2,2))
# plot(train_x_data[1,,,1:3] %>% imager::as.cimg())
# plot(train_x_data[1,,,4:6] %>% imager::as.cimg())
# plot(train_y_data[1,,,11] %>% imager::as.cimg())
# plot(train_y_data[1,,,8] %>% imager::as.cimg())

###---------------------------Model Definition---------------------------------------
# Create u-net architecture according to predefined values and other values
# passed by the FLAGS (for hyperparameter exploration)
model <- unet(input_shape = c(img_width, img_height, channels),
              num_classes = n_classes,
              dropout = FLAGS$dropout,
              filters = FLAGS$filters_firstlayer,
              num_layers = FLAGS$num_layers,
              output_activation = activation_func_out)

# Let's see the model
summary(model)

# define f1score formula
f1score <- custom_metric("f1score", function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

# Compile model, set loss function, as well as the metrics that should be supervised in
# each epoch
model %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$learn_rate),
  loss = "categorical_crossentropy",
  metrics = list(f1score, "categorical_accuracy")
)

# Create an early stopping rule
early_stopping <- callback_early_stopping(monitor = 'val_loss', 
                                          min_delta = 0.01,
                                          mode = "min",
                                          patience = 10,
                                          restore_best_weights = T)

##-------------------------------------Run Model-------------------------------

# Fit model
history <- model %>% fit(
  #Model and inputs
  #X as independent vars
  x = train_x_data,
  y = train_y_data,
  # Remember we can use validation split also, but right now, prefer not, to force to leave out certain images from the trainaing (as we are augumenting images)
  validation_data = list(test_x_data, test_y_data),
  #Epochs
  epochs = FLAGS$epochs,
  #steps_per_epoch = steps_per_epoch,
  #Batch_size
  batch_size = FLAGS$batch_size,
  shuffle = T,
  #Verbose or not
  verbose = 1,
  callbacks = list(early_stopping)
)

# Save model
save_model_hdf5(model, paste0("U128model",imagery,
                              "filters", FLAGS$filters_firstlayer,
                              "Epochs",FLAGS$epochs,
                              "layers", FLAGS$num_layers,
                              "dropout", FLAGS$dropout,
                              "_lr",FLAGS$learn_rate,
                              "_adam",
                              "_2020-12-01",".h5"))

# Plot history
plot(history)

# See scores
score <- model %>% evaluate(
  test_x_data, test_y_data,
  verbose = 0
)

cat('Test loss:', score$loss, '\n')
cat('Test accuracy:', score$acc, '\n')
cat('Test f1score:', score$f1score, '\n')
cat('Train loss:', score$loss, '\n')
cat('Train accuracy:', score$acc, '\n')
cat('Train f1score:', score$f1score, '\n')

# Remove all objects
rm(list=ls())