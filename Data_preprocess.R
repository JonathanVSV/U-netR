library(raster)
library(rray)
library(Hmisc)
library(imager)
library(reticulate)

# Number of squared samples (windows of x by x pixels) for which you have training data
num_imagenes <- 23

# Use numbers to name training data of image samples (reflectance and backscatter values) 
# and image masks (land use land cover [LULC] categories)
nombres_mask <- seq(1,23,1)

# Number of LULC classes
n_classes <- 11
# Size of windows in pixels (training data)
img_width <- 256
img_height <- 256
# Size of windows in pixels (augumented training data)
img_width_exp <- 128
img_height_exp <- 128
# Number of bands of the input image
channels <- 6

# Number of 128pix-squares that are going to be obtained per 256 pix-squares
# In this case we are going to use 25 (5 x 5 128 x 128 pix) using a 32 pix offset
# This is cropping the image as 1-129, 1-129; 33-161, 1-129; 65-193, 1-129; 97-225, 1-129; 129-256, 1-129 
num_squares <- 5^2
probs_4crops <- 1 / (sqrt(num_squares)-1) 
# Number of mirrored images per 128pix-image
num_mirrors <- 2

# ---------------Read Training Info----------------------------------------------------
# Read files for rectangles for which you have training data (land cover / land use)
cuads <- list.files("cuads",
                    "*.shp$",
                    full.names = T)
cuads <- lapply(cuads, shapefile)

# Locations of image files
optic_names <- "Sentinel-2_im.tif"
radar_names <- "Sentinel-1_im.tif"

# Read images as stacks
optic <- stack(optic_names)
radar <- stack(radar_names)

# Input data that is going to be pre processed to further feed the U-net
# Features: Stack optic 4 bands and 2 radar bands
optic <- stack(optic,radar[[1:2]])

# Labels: Read files of manually classificated areas 
masks_list <- list.files("masks",
                         "*.shp$",
                         full.names = T)
masks <- lapply(masks_list, shapefile)

# ---------------Crop Images----------------------------------------------------
# Crop images (features data) to extents of shapefiles
cropped_im <- lapply(cuads, function(x){
  temp <- crop(optic, x)
})

# Crop class info (labels data) to extents of shapefiles
cropped_masks <- lapply(1:length(masks), function(i){
  temp <- crop(masks[[i]], cuads[[i]])
  temp@data$id = as.integer(temp@data$id)
  rasterize(temp, cropped_im[[i]],
            field = "id",
            background = 0)
  
})

# Create an empty list to save cropped class info
y_rast_list <- vector(mode = "list", 
                      length = length(cropped_masks))

# Change masks to binary representations
# List with two dimensions, first index is number of image
# Second index is type of cover mask
for (j in 1:length(cropped_masks)){
  y_rast_list[[j]] <- stack(lapply(1:n_classes, function(i){
    cropped_masks[[j]] == i
  }))
}

# Plot
plot(cropped_im[[7]])
plot(y_rast_list[[7]])

# ---------------Convert info to arrays----------------------------------------------------
# Change image (features data) to type array
cropped_im_matr <- lapply(cropped_im, as.array)
# Change class info (labels data) to type array
cropped_mask_matr <- lapply(y_rast_list, as.array)

# Create empty arrays to fill it with the previous info
train_x_data<-array(0,
                    dim = c(length(cropped_im_matr),
                            img_height,
                            img_width,
                            channels))
train_y_data<-array(0,
                    dim = c(length(cropped_im_matr),
                            img_height,
                            img_width,
                            n_classes))

# Fill the arrays
for(i in 1:length(cropped_im_matr)) {
  train_x_data[i,1:img_width,1:img_height,1:channels] <- cropped_im_matr[[i]]
}

for(i in 1:length(cropped_mask_matr)) {
    train_y_data[i,1:img_width,1:img_height,1:n_classes] <- cropped_mask_matr[[i]]
}

# ---------------Create 128x128 px arrays----------------------------------------------------
# Create 128 x 128 arrays
train_x_data_128<-array(0,
                    dim = c(length(cropped_im_matr) * num_squares,
                            img_height_exp,
                            img_width_exp,
                            channels))
train_y_data_128<-array(0,
                        dim = c(length(cropped_mask_matr) * num_squares,
                                img_height_exp,
                                img_width_exp,
                                n_classes))

# Vectors with starting positions to make 128 x 128 pix squares
temp <- unique(quantile(seq(1,img_width_exp+1), probs = seq(0, 1, probs_4crops)))
# The valuse that are going to be used to crop
crops <- c(temp,img_width_exp+1)
crops

# Value to sum to each of the previous values
sum_pix <- img_height_exp - 1
aux <- 1

# Fill the empty arrays with the info, 256 x 256 images get cropped in the 128 x 128 px
for(i in 1:dim(train_x_data)[1]) {
  for(j in 1:length(crops)){
    for(k in 1:length(crops)){
      train_x_data_128[aux,,,] <- train_x_data[i,crops[j]:(crops[j]+sum_pix),
                                                 crops[k]:(crops[k]+sum_pix),
                                                 1:channels]
      aux <- aux + 1
    }
  }
}

aux <- 1
for(i in 1:dim(train_y_data)[1]) {
  for(j in 1:length(crops)){
    for(k in 1:length(crops)){
      train_y_data_128[aux,,,] <- train_y_data[i,crops[j]:(crops[j]+sum_pix),
                                               crops[k]:(crops[k]+sum_pix),
                                               1:n_classes]
      aux <- aux + 1
    }
  }
}

# ---------------Mirrored Images----------------------------------------------------
# Get number of total images (original + 2 mirrors)
multip_im <- num_mirrors + 1

# Create 128 x 128 empty arrays, self + 2 mirrors
train_x_data_final<-array(0,
                        dim = c(dim(train_x_data_128)[1] * multip_im,
                                img_height_exp,
                                img_width_exp,
                                channels))
train_y_data_final<-array(0,
                        dim = c(dim(train_y_data_128)[1] * multip_im,
                                img_height_exp,
                                img_width_exp,
                                n_classes))


aux <- 1

# Create mirrored images and fill the empty arrays
for(i in 1:dim(train_x_data_128)[1]) {
  train_x_data_final[aux,,,] <- train_x_data_128[i,,,]
  aux = aux + 1
  train_x_data_final[aux,,,] <- rray_flip(train_x_data_128[i,,,],axis = 1)
  aux = aux + 1
  train_x_data_final[aux,,,] <- rray_flip(train_x_data_128[i,,,],axis = 2)
  aux = aux + 1
}

aux <- 1

for(i in 1:dim(train_y_data_128)[1]) {
  train_y_data_final[aux,,,] <- train_y_data_128[i,,,]
  aux = aux + 1
  train_y_data_final[aux,,,] <- rray_flip(train_y_data_128[i,,,],axis = 1)
  aux = aux + 1
  train_y_data_final[aux,,,] <- rray_flip(train_y_data_128[i,,,],axis = 2)
  aux = aux + 1
}

# Check if everything is ok
par(mfrow = c(2,2))
plot(train_x_data_final[207,1:128,1:128,1:3] %>% as.cimg())
plot(train_y_data_final[207,1:128,1:128,6] %>% as.cimg())
plot(train_y_data_final[207,1:128,1:128,11] %>% as.cimg())

# ---------------Normalize Images----------------------------------------------------
# Before exporting all the data, let's calculate per band mean and SD to normalize the data
mean_x <- apply(train_x_data_final, 4, mean)
sd_x <- apply(train_x_data_final, 4, sd)

# Normalize data (only features data)
for(i in 1:dim(train_x_data_final)[1]){
  for(j in 1:dim(train_x_data_final)[4]){
    train_x_data_final[i,,,j] <- (train_x_data_final[i,,,j] - mean_x[j]) / sd_x[j]
  }
}

#---------------Image checks----------------------------------------------------
#Check min and max values
apply(train_x_data_final, 4, min)
apply(train_x_data_final, 4, max)

# Check NA, NaN, Inf, -Inf; which might cause the loss value to fix in Nan, when training the U-net 
which(apply(train_x_data_final, 1, is.nan))
which(apply(train_x_data_final, 1, is.na))
which(apply(train_x_data_final, 1, function(x) x == Inf))
which(apply(train_x_data_final, 1, function(x) x == -Inf))

which(apply(train_y_data_final, 1, is.nan))
which(apply(train_y_data_final, 1, is.na))
which(apply(train_y_data_final, 1, function(x) x == Inf))
which(apply(train_y_data_final, 1, function(x) x == -Inf))

# Check normalized images
plot(train_x_data_final[500,1:128,1:128,3:1] %>% as.cimg())
plot(train_y_data_final[500,1:128,1:128,6] %>% as.cimg())
plot(train_y_data_final[500,1:128,1:128,4] %>% as.cimg())

# ---------------Test and Train sets----------------------------------------------------
# Create train and test sets
# Number of images (128 x 128 px) that come from the same 256 x 256 image
imgs_share <- (multip_im * num_squares) - 1

# Images that are going to be saved as test data; and thus, removed from the training data
# With this procedure we completely omit all the augumented images from a single
# 256 x 256 px square.
testers <- c(1+imgs_share,imgs_share*10+1)

# Define test and training data
test_x_data_final <- train_x_data_final[c(1:testers[1],testers[2]:(testers[2]+imgs_share)),,,]
test_y_data_final <- train_y_data_final[c(1:testers[1],testers[2]:(testers[2]+imgs_share)),,,]

train_x_data_final <- train_x_data_final[-c(1:testers[1],testers[2]:(testers[2]+imgs_share)),,,]
train_y_data_final <- train_y_data_final[-c(1:testers[1],testers[2]:(testers[2]+imgs_share)),,,]

# Plot images to check that everything is ok
plot(train_x_data_final[1,1:img_height_exp,1:img_width_exp,3:1] %>% as.cimg())
plot(train_y_data_final[1,1:img_height_exp,1:img_width_exp,8] %>% as.cimg())
plot(train_y_data_final[1,1:img_height_exp,1:img_width_exp,11] %>% as.cimg())

# One option is to save files as rdf
# Save as rdf files
# saveRDS(train_x_data_final, file = "train_x_data.rds", ascii = F)
# saveRDS(train_y_data_final, file = "train_y_data.rds", ascii = F)
# 
# saveRDS(test_x_data_final, file = "test_x_data.rds", ascii = F)
# saveRDS(test_y_data_final, file = "test_y_data.rds", ascii = F)

# ---------------Export to npz----------------------------------------------------
# Other option: saving files as numpy zip files
# This option was prefered as a single file can contain both training and test data
np <- import("numpy")

np$savez("Img_Preprocess/train_test_sets.npz", 
         x_train = train_x_data_final, 
         y_train = train_y_data_final, 
         x_test = test_x_data_final, 
         y_test = test_y_data_final)


