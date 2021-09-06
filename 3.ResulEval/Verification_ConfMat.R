library(unet)
library(keras)
library(reticulate)
library(tfruns)
library(raster)
library(rgdal)
library(rasterVis)
library(tidyverse)
library(yardstick)

# Best model
model_location <- paste0("U128modelOpticRadarfilters64Epochs50layers3dropout0.5_lr1e-04_adam_2021-03-01.h5")

# Imagen
imagery <- "OpticRadar"
# Label data
n_classes <- 12
# Number of bands
channels <- 6

# Unet parameters, can be extracted from the hyperparameter exploration csv
batch_size <- 16
learn_rate <- 1e-4
epochs <- 50
dropout <- 0.5
filters_firstlayer <- 64
num_layers <- 3

# Los demas params
activation_func_out <- "softmax"
# 
# Input image dimensions
img_width <- 128
img_height <- 128
# 
# Predicted image dimensions
img_width_pred <- 128
img_height_pred <- 128

# Training set location
train_image_files_path <- file.path(paste0("training"))

# Numpy zip file option
np <- import("numpy")

# Load npz
npz2 <- np$load(paste0(train_image_files_path,"/","LULC_",input_type,"_Augment9perim_samples",".npz"))

# See files
npz2$files

# Load test data only
test_x_data <- npz2$f[["x_test"]]
test_y_data <- npz2$f[["y_test"]]

par(mfrow = c(2,2))
raster::plotRGB(raster::stack(raster::raster(as.matrix(test_x_data[1,,,3])),
                              raster::raster(as.matrix(test_x_data[1,,,2])),
                              raster::raster(as.matrix(test_x_data[1,,,1]))),
                r = 1, 
                g = 2, 
                b = 3,
                scale=800, 
                stretch = "lin")
raster::plotRGB(raster::stack(raster::raster(as.matrix(test_x_data[40,,,3])),
                              raster::raster(as.matrix(test_x_data[40,,,2])),
                              raster::raster(as.matrix(test_x_data[40,,,1]))),
                r = 1, 
                g = 2, 
                b = 3,
                scale=800, 
                stretch = "lin")
raster::plotRGB(raster::stack(raster::raster(as.matrix(test_x_data[67,,,3])),
                              raster::raster(as.matrix(test_x_data[67,,,2])),
                              raster::raster(as.matrix(test_x_data[67,,,1]))),
                r = 1, 
                g = 2, 
                b = 3,
                scale=800, 
                stretch = "lin")
plot(test_y_data[220,,,10] %>% imager::as.cimg())

###---------------------------Model Definition---------------------------------------
# Define f1score
f1score <- custom_metric("f1score", function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

##-------------------Result Evaluation-----------------------------
# load the best model (already trained)
model <- load_model_hdf5(model_location,
                         custom_objects = c("f1score" = f1score))

# see score
score <- model %>% evaluate(
  test_x_data, test_y_data,
  verbose = 1
)

score

# Make predictions over test set to calculate the error matrix and f1-scores
accuracy <- vector(length = dim(test_x_data)[1])
conf_matrix <- vector(length = dim(test_x_data)[1], mode = "list")

#Classes 
  classes <- c("1","2","3","4",
               "5","6","7","8","9","10",
               "11", "12")
  classes_num <- c(seq(1, n_classes, 1))

# Create empty lists that are going to be filled in the next loop
plots_list <- vector(length = dim(test_x_data)[1] * 3, mode = "list")
# PRedicted images
rast_pred_list <- vector(length = dim(test_x_data)[1], mode = "list")
# Ground truth images
rast_gt_list <- vector(length = dim(test_x_data)[1], mode = "list")
# Probabilities raster
prob_list <- vector(length = dim(test_x_data)[1], mode = "list")

# Loop to fill lists
for(j in 1:dim(test_x_data)[1]){
  image_real<-array_reshape(test_x_data[j,,,1:channels],c(1,img_width_pred,img_height_pred,channels))
  
  result_pred <- predict(
    object = model,
    batch_size = batch_size,
    x = image_real,
    steps = 10,
    verbose = 1)
  
  image_class <- array_reshape(test_y_data[j,,,1:n_classes],c(1,img_width_pred,img_height_pred,n_classes))
  
  # predicted image
  image_pred <- array_reshape(result_pred[1,,,1:n_classes],c(1,img_width_pred,img_height_pred,n_classes))
  
  # Get max probability value by cell in all the classes, i.e., dim 3
  maxval_bycell <- apply(image_pred[1,,,] , c(1,2) , 
                         function(x) 
                           ifelse(all(is.na(x)), NA, max(x, na.rm = TRUE))) 
  
  resul2 <- lapply(1:n_classes, function(i){
    # Set class as the pixel that match the max prob (remember this comes from softmax classificator 0-1)
    temp <- image_pred[1,,,i] == maxval_bycell
    # temp <- image_pred[1,,,i] >= 0.5
    # temp2 <- image_pred[1,,,i] < 0.5 & image_pred[1,,,i] >= 0.1
    temp[temp == TRUE] <- i
    # temp[temp2 == TRUE] <- n_classes + 1
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  
  # reduce
  resul2 <- Reduce('+', resul2)
  
  resul <- lapply(1:n_classes, function(i){
    temp <- image_class[1,,,i]==1
    temp[temp == TRUE] <- i
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  
  # reduce
  resul_class <- Reduce('+', resul)
  
  # To see spatial distributino of errors
  # rast_list[[j]] <- resul_class - resul2
  rast_pred_list[[j]] <- resul2
  prob_list[[j]] <- maxval_bycell
  rast_gt_list[[j]] <- resul_class
  
  # Confusion matrix, completing missing levels so that the diagonal corresponds to 
  # correctly predicted pixels
  # Resul2 is predicted, resul_class is ground truth
  confusion_matrix<- table(factor(resul2,classes_num), factor(resul_class, classes_num))
  conf_matrix[[j]] <- confusion_matrix
  #Calculate overall precision
  accuracy[j] <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  if(imagery == "Radar"){
    temp <- stack(raster(as.matrix(image_real[1,,,2])),
                  raster(as.matrix(image_real[1,,,1])))
  }else{
    temp <- stack(raster(as.matrix(image_real[1,,,3])),
                  raster(as.matrix(image_real[1,,,2])),
                  raster(as.matrix(image_real[1,,,1])))
  }
}

# check distribution of errors
# rast_list <- lapply(rast_list, function(x) ifelse(x == 0, 0, 1))
plot(as.raster(rast_pred_list[[1]]/13))

# Check areas with low probability of belonging to each class
plot(as.raster(prob_list[[1]]<=0.5))
plot(as.raster(probs[[200]]/13))

# Total mean accuracy
mean(accuracy)

# 128 x 128 version tiene accuracy como de 0.70
# 256 x 256 version tiene accuracy como de 0.76
# Get sum of all confusion matrices
total_conf_mat <- Reduce('+', conf_matrix)
colnames(total_conf_mat) <- classes
rownames(total_conf_mat) <- classes
total_conf_mat

# Get product and user accuracy
users_accuracy <- sapply(1:nrow(total_conf_mat), function(i){
  total_conf_mat[i,i] / sum( total_conf_mat[i,])
})
names(users_accuracy) <- colnames(total_conf_mat)
users_accuracy

producers_accuracy <- sapply(1:ncol(total_conf_mat), function(i){
  total_conf_mat[i,i] / sum( total_conf_mat[,i])
})
names(producers_accuracy) <- row.names(total_conf_mat)
producers_accuracy

overall_acc <- sum(diag(as.matrix(total_conf_mat))) / sum(as.matrix(total_conf_mat))

ncol(total_conf_mat)
total_conf_mat <- cbind(total_conf_mat,users_accuracy)
total_conf_mat <- rbind(total_conf_mat,c(producers_accuracy,overall_acc))

write.csv(total_conf_mat, paste0("ConfusionMatrix_TestData_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam","_2021-03-15",".csv"))
write.csv(mean(accuracy), paste0("MeanAccuracy_TestData_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam","_2021-03-15",".csv"))

#--------------------------f1sccore-------------------------------------------------------------

# Prepare data
total_conf_matrix <- read.csv(paste0("ConfusionMatrix_TestData_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam","_2021-03-15",".csv"))

# Remove prod user accuracy and class names
# For OpticRadar and Optic
total_conf_mat <- total_conf_matrix [-13,-c(1,14)]

## ----------------------------Yardstick evaluation-------------------------------------------------------

# Remake simple counts from confusion matrix
# OpticRadar and Optic
colnames(total_conf_mat) <- 1:12
# Radar
# colnames(total_conf_mat) <- 1:10

temp <- total_conf_mat %>%
  as_tibble() %>%
  rownames_to_column() %>%
  pivot_longer(cols = -rowname,
               names_to = "Class_pred",
               values_to = "count") %>%
  rename("Class_true" = "rowname")

temp_gt_pred <- lapply(1:nrow(temp), function(i){
  data.frame(gt = rep(temp$Class_true[i], temp$count[i]), 
             pred = rep(temp$Class_pred[i], temp$count[i]))
})
temp_gt_pred <- bind_rows(temp_gt_pred)

estimates_keras_tbl <- tibble(
  truth      = factor(temp_gt_pred[,1], levels = 1:12),
  estimate   = factor(temp_gt_pred[,2], levels = 1:12),
  # class_prob = as.numeric(unlist(temp_probs))
)

# Confusion Table
estimates_keras_tbl %>% 
  conf_mat(truth, estimate)

# Accuracy
estimates_keras_tbl %>% 
  metrics(truth, estimate)

## F1 score, Esto da F1 score de 0.62 (macro)
# micro = 0.76; macro_weighted = 0.77
estimates_keras_tbl %>% 
  f_meas(truth, 
         estimate, 
         estimator = "macro",
         beta = 1)