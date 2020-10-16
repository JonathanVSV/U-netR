library(keras)
library(tidyverse)
library(raster)
library(rgdal)
library(reticulate)
library(rasterVis)

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
npz2 <- np$load(paste0(train_image_files_path,"/","train_test_sets.npz"))
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

# ------------------------------Load previously saved model-----------------------------
# Define dice to be able to load model
dice <- custom_metric("dice", function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})


model <- load_model_hdf5( paste0("U128model",imagery,
                                 "filters", filters_firstlayer,
                                 "Epochs",epochs,
                                 "layers",num_layers,
                                 "dropout", dropout,
                                 "_lr",learn_rate,
                                 "_adam",".h5"),
                        custom_objects = c("dice" = dice))


#----------------------Predict images from test set-------------------------------------

# Create an empty vector to calculate the overall accuracy for each predicted image
accuracy <- vector(length = dim(test_x_data)[1])
# Create an empty list to save the confusion matrix for each predicted image
conf_matrix <- vector(length = dim(test_x_data)[1], mode = "list")

# Names of the classes that we are using
classes <- c("NoData","PlantacionesMaduras","BosqueMaduro","BosqueSecundario","Caminos",
             "PlantacionesJovenes","Suelo","Cultivos/Pastizales","Agua","Nubes","Sombras",
             "AsentamientoHumano")

# Create two color pallettes 1) to plot classified images and
# 2) plot probability of each pixel of belonging to the predicted class
pal <- colorRampPalette(c("red","green"))
pal2 <- colorRampPalette(c("red","orange"))

# Para exportar los plots nada más modificar el ciclo y poner anter y después del ciclo el jpeg y dev.off
aux <- 1.1

every3 <- seq(1, dim(test_x_data)[1], 3)
end3 <- seq(3, dim(test_x_data)[1], 3)

# Predict each test image and calculate its confusion matrix
for(j in 1:dim(test_x_data)[1]){
  
  # Get features data (images)
  image_real<-array_reshape(test_x_data[j,,,1:channels],c(1,img_width_pred,img_height_pred,channels))
  
  # Predict class (y_pred)
  result_pred <- predict(
    object = model,
    batch_size = batch_size,
    x = image_real,
    steps = 10,
    verbose = 1)
   
  # Get label data (classess)
  image_class <- array_reshape(test_y_data[j,,,1:n_classes],c(1,img_width_pred,img_height_pred,n_classes))
  
  # Get predicted image
  image_pred <- array_reshape(result_pred[1,,,1:n_classes],c(1,img_width_pred,img_height_pred,n_classes))
  
  rast_list[[j]] <- image_pred
  
  # Get max probability value by cell in all the classes, i.e., dim 3
  maxval_bycell <- apply(image_pred[1,,,] , c(1,2) , 
                         function(x) 
                           ifelse(all(is.na(x)), NA, max(x, na.rm = TRUE))) 
  
  # Set class as the pixel that match the max prob (remember this comes from softmax classificator 0-1)
  # This is y_pred
  resul2 <- lapply(1:n_classes, function(i){
    temp <- image_pred[1,,,i] == maxval_bycell
    temp[temp == TRUE] <- i
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  
  # Reduce resuls to transform binary bands to multiband image
  resul2 <- Reduce('+', resul2)
  
  # Set class as the pixel that match 1
  # This is y
  resul <- lapply(1:n_classes, function(i){
    temp <- image_class[1,,,i]==1
    temp[temp == TRUE] <- i
    temp[temp == FALSE] <- 0
    as.matrix(temp)
    
  })
  # Reduce resul to transform binary bands to multiband image
  resul_class <- Reduce('+', resul)
  
  # Confusion matrix, we need to complete missing levels so that the diagonal corresponds to 
  # correctly predicted pixels
  confusion_matrix<- table(factor(resul2,seq(0,11,1)), factor(resul_class, seq(0,11,1)))
  conf_matrix[[j]] <- confusion_matrix
  
  #Calculate overall precision
  accuracy[j]<-sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  temp <- stack(raster(as.matrix(image_real[1,,,3])),
                raster(as.matrix(image_real[1,,,2])),
                raster(as.matrix(image_real[1,,,1])))
  
  # Export a jpeg of every 3 samples
  if(j %in% every3) {
    jpeg(paste0("Predicted_images", j, ".jpeg"),
         width = 30,
         height = 21,
         units = "cm",
         quality = 100,
         res = 300)
    par(mfrow = c(3, 5),
        mai = c(0.1,0.1,0.1,0.1),
        pin = c(3,3),
        bty = 'n')
  }
  
  # Plot test data image (x_test)
  plotRGB(temp, r = 1, g = 2, b = 3,scale=800, stretch = "lin")
  
  # Plot test class info (y_test)
  plot(raster(resul_class/12),
       col=rainbow(11),
       breaks = seq(0,1,0.1),
       xaxt = "n",
       yaxt = "n",
       axes=FALSE,
       legend = F)
  
  # Plot test class perdicted (y_pred_test)
  plot(raster(resul2/12),
       col=rainbow(11),
       breaks = seq(0,1,0.1),
       xaxt = "n",
       yaxt = "n",
       axes=FALSE,
       legend = F)
  
  # Plot probability of each pixel of being of the assigned class
  plot(raster(maxval_bycell),
       col = pal(10),
       breaks = seq(0,1,0.1),
       xaxt = "n",
       yaxt = "n",
       axes=FALSE,
       legend = T)
  # Plot probability of each pixel of being of the assigned class >= 0.5
  plot(raster(maxval_bycell >= 0.5),
       col = pal2(6),
       breaks = seq(0,0.5,0.1),
       xaxt = "n",
       yaxt = "n",
       axes=FALSE,
       legend = T)

  if(j %in% end3) {
    dev.off()
  }
  # Pplot to addlegend
  # plot(raster(resul2/12),
  #      col=rainbow(11),
  #      breaks = seq(0,1,0.1),
  #      xaxt = "n",
  #      yaxt = "n",
  #      axes=FALSE,
  #      legend = F)
  # legend(x=-1,y=aux,
  #        legend = classes[-1],
  #        fill=rainbow(12)[1:12],
  #        cex = 0.9,
  #        box.col = "transparent")
  aux <- aux + 1.1
  
  aux <- ifelse(aux >= 6, 1.1, aux)
  
  
}

# Total mean accuracy
mean(accuracy)

# Get sum of all confusion matrices
total_conf_mat <- Reduce('+', conf_matrix)
colnames(total_conf_mat) <- classes
rownames(total_conf_mat) <- classes
total_conf_mat

# Write the complete confusion matrix and accuracy
write.csv(total_conf_mat, paste0("ConfusionMatrix_TestData_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam","_2019-10-06",".csv"))
write.csv(mean(accuracy), paste0("MeanAccuracy_TestData_","U128model",imagery,"Epochs",epochs,"_lr",learn_rate,"_adam","_2019-10-06",".csv"))


# Get product and user accuracy
users_accuracy <- sapply(1:nrow(total_conf_mat), function(i){
  total_conf_mat[i,i] / sum( total_conf_mat[i,])
})
names(users_accuracy) <- colnames(total_conf_mat)
users_accuracy

producers_accuracy <- sapply(1:nrow(total_conf_mat), function(i){
  total_conf_mat[i,i] / sum( total_conf_mat[,i])
})
names(producers_accuracy) <- row.names(total_conf_mat)
producers_accuracy

####---------------------Visualize intermediate activation layers------------------------------------------
model
# Extracts the outputs of the top 2 layers:
# Cargar ciertas capas de activación del vgg16
block2conv <- keras_model(inputs = model$input,
                          outputs = get_layer(model, 'conv2d_7')$output)

# Returns a list of five arrays: one array per layer activation
# Esto no sirve si utilizas la bateria
activations <- block2conv %>% predict(test_x_data[1:2,,,])
par(mfrow = c(10, 4), mai = c(0.1,0.1,0.1,0.1), bty = 'n')

for(i in 1:39)
{
  temp <- activations[1,1:16,1:16, i]
  plot(as.raster(temp/max(temp)))
}


dim(first_layer_activation)

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, 
        col = terrain.colors(12))
}

pdf("First_layer_activation.pdf",
    width=15,
    height = 12)
par(mfrow = c(8, 6),mai = rep_len(0.02, 4))
for(i in 1:48){
  plot_channel(first_layer_activation[1,,,i])
}
dev.off()

pdf("Second_layer_activation.pdf",
    width=15,
    height = 12)
par(mfrow = c(6, 6),mai = rep_len(0.02, 4))
for(i in 1:32){
  plot_channel(second_layer_activation[1,,,i])
}
dev.off()

pdf("Third_layer_activation.pdf",
    width=15,
    height = 12)
par(mfrow = c(1, 1))
for(i in 1:1){
  plot_channel(third_layer_activation[1,,,i])
}
dev.off()