# Load libraries
  library(raster)
  library(rray)
  library(Hmisc)
  library(imager)
  library(reticulate)
  library(sf)
  library(tidyverse)
  
  # User defined variables
  input_type <- "OpticRadar"
  
  # Number of squared samples (windows of x by x pixels) for which you have training data
  num_imagenes <- 33
  
  # Use numbers to name training data of image samples (reflectance and backscatter values) 
  # and image masks (land use land cover [LULC] categories)
  nombres_mask <- seq(1,num_imagenes,1)
  
  # Name of the field in a shapefile that contains the manual LULC classification
  field <- "newid"
  
  # Number of LULC classes (including shadows and clouds that are going to be ignored)
  n_classes <- 12 
  
  # Number of bands of the input image
  # OpticRadar
  channels <- 6
  # Optic
  # channels <- 4
  # Radar
  # channels <- 2
  
  # Size of windows in pixels (training data)
  img_width <- 256
  img_height <- 256
  # Size of windows in pixels (augumented training data)
  img_width_exp <- 128
  img_height_exp <- 128
  
  
  # Number of 128pix-squares that are going to be obtained per 256 pix-squares
  # In this case we are going to use 27 (9 x 3 128 x 128 pix squares) using a 64 pix offset
  # This is cropping the image as 1-129, 1-129; 65-161, 1-129; 129-193, 1-129 etc.
  num_squares <- 3^2
  probs_4crops <- 1 / (sqrt(num_squares)-1) 
  # Number of mirrored images per 128pix-image
  num_mirrors <- 2
  
  # ---------------Read Training Info----------------------------------------------------
  # Read files for rectangles for which you have training data (land cover / land use)
  cuads <- paste0("areas4manualLULC",
                  ".shp")
  cuads <- st_read(cuads)
  
  # Locations of image files
  optic_names <- "4BSentinel-2img.tif"
  
  # Esta versión ya está leyendo los datos originales de backscatter en gamma0
  radar_names <- "2BSentinel-1img.tif"
  
  # Read images as stacks
  optic <- stack(optic_names)
  radar <- stack(radar_names)
  
  # Crop to have the same extent
  radar <- crop(radar, optic)
  
  # Input data that is going to be pre processed to further feed the U-net
  # Features: Stack optic 4 bands and 2 radar bands
  optic <- stack(optic,radar[[1:2]])
  
  # Just optic
  # optic <- optic
  
  # Radar only info
  # optic <- radar[[1:2]]
  
  # Labels: Read files of manually classificated areas 
  # Optic labeles
  masks_list <- paste0("manualLULCinfo",".shp")
  # # Radar labels
  # masks_list <- paste0("D:/Drive/Jonathan_trabaggio/Doctorado/GeoInfo/TiposVeg_points/4SingleMaskRadar/new_corrected/SinglePolys_4array_corrected",
  # ".shp")
  
  masks <- st_read(masks_list)
  
  print("Check number of classes")
  
  #Check number of classes
  unique(masks$DN)[order(unique(masks$DN))]
    
  # ---------------Crop Images----------------------------------------------------
  # Crop images (features data) to extents of shapefiles
  cropped_im <- lapply(1:dim(cuads)[1], function(i){
    singlecuad <- cuads %>%
      slice(i)
    temp <- crop(optic, singlecuad)
  })
  
  # Crop class info (labels data) to extents of shapefiles
  cropped_masks <- lapply(1:dim(cuads)[1], function(i){
    temp <- st_crop(masks, cuads %>%
                   slice(i))
    # temp@data$id = as.integer(temp@data$id)
    rasterize(temp, cropped_im[[i]],

              field = field,
              background = 0)
    
  })
  
  # Plot images
  plot(cropped_im[[1]])
  plot(cropped_masks[[1]])
  
  # Remove optic and masks that are no longer going to be used
  rm(optic, masks)
  
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

# Remove objects that are no longer used
rm(cropped_im,y_rast_list,cropped_masks)

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

# Remove unused objects
rm(cropped_im_matr, cropped_mask_matr)

# Vectors with starting positions to make 128 x 128 pix squares
temp <- unique(quantile(seq(1,img_width_exp+1), probs = seq(0, 1, probs_4crops)))
# The valuse that are going to be used to crop
crops <- c(temp)
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
rm(train_x_data,train_y_data)

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

rm(train_x_data_128, train_y_data_128)

# Check if everything is ok
par(mfrow = c(2,2))
plot(train_x_data_final[207,1:128,1:128,1:3] %>% as.cimg())
plot(train_y_data_final[207,1:128,1:128,6] %>% as.cimg())
plot(train_y_data_final[207,1:128,1:128,11] %>% as.cimg())

# ---------------Normalize Images----------------------------------------------------
# Before exporting all the data, let's calculate per band mean and SD to normalize the data
mean_x <- apply(train_x_data_final, 4, mean)
sd_x <- apply(train_x_data_final, 4, sd)

# Write means and sd to csv files that are afterwards going to be used
# in the scripts that prepares the prediction over the complete image
write.csv(mean_x, paste0("Mean_x",input_type,".csv"), row.names = F)
write.csv(sd_x, paste0("sd_x",input_type,".csv"), row.names = F)

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
# which(apply(train_x_data_final, 1, is.nan))
# which(apply(train_x_data_final, 1, is.na))
# which(apply(train_x_data_final, 1, function(x) x == Inf))
# which(apply(train_x_data_final, 1, function(x) x == -Inf))
# 
# which(apply(train_y_data_final, 1, is.nan))
# which(apply(train_y_data_final, 1, is.na))
# which(apply(train_y_data_final, 1, function(x) x == Inf))
# which(apply(train_y_data_final, 1, function(x) x == -Inf))

# Check normalized images
plot(train_x_data_final[500,1:128,1:128,3:1] %>% as.cimg())
plot(train_y_data_final[500,1:128,1:128,6] %>% as.cimg())
plot(train_y_data_final[500,1:128,1:128,4] %>% as.cimg())

# ---------------Test and Train sets----------------------------------------------------
# Create train and test sets
# Number of images (128 x 128 px) that come from the same 256 x 256 image
imgs_share <- (multip_im * num_squares)

# Images that are going to be saved as test data; and thus, removed from the training data
# With this procedure we completely omit all the augumented images from a single
# 256 x 256 px square.
testers <- seq(1, dim(train_x_data_final)[1]+1, imgs_share)

# Define test and training data
# Here it was made manually, but can be done at random
test_x_data_final <- train_x_data_final[c(testers[2]:(testers[3]-1),
                                          testers[4]:(testers[5]-1),
                                          testers[5]:(testers[6]-1),
                                          testers[7]:(testers[8]-1),
                                          testers[9]:(testers[10]-1),
                                          testers[17]:(testers[18]-1),
                                          testers[19]:(testers[20]-1),
                                          testers[26]:(testers[27]-1),
                                          testers[29]:(testers[30]-1),
                                          testers[33]:(testers[34]-1)),,,]
test_y_data_final <- train_y_data_final[c(testers[2]:(testers[3]-1),
                                          testers[4]:(testers[5]-1),
                                          testers[5]:(testers[6]-1),
                                          testers[7]:(testers[8]-1),
                                          testers[9]:(testers[10]-1),
                                          testers[17]:(testers[18]-1),
                                          testers[19]:(testers[20]-1),
                                          testers[26]:(testers[27]-1),
                                          testers[29]:(testers[30]-1),
                                          testers[33]:(testers[34]-1)),,,]

train_x_data_final <- train_x_data_final[-c(testers[2]:(testers[3]-1),
                                            testers[4]:(testers[5]-1),
                                            testers[5]:(testers[6]-1),
                                            testers[7]:(testers[8]-1),
                                            testers[9]:(testers[10]-1),
                                            testers[17]:(testers[18]-1),
                                            testers[19]:(testers[20]-1),
                                            testers[26]:(testers[27]-1),
                                            testers[29]:(testers[30]-1),
                                            testers[33]:(testers[34]-1)),,,]
train_y_data_final <- train_y_data_final[-c(testers[2]:(testers[3]-1),
                                            testers[4]:(testers[5]-1),
                                            testers[5]:(testers[6]-1),
                                            testers[7]:(testers[8]-1),
                                            testers[9]:(testers[10]-1),
                                            testers[17]:(testers[18]-1),
                                            testers[19]:(testers[20]-1),
                                            testers[26]:(testers[27]-1),
                                            testers[29]:(testers[30]-1),
                                            testers[33]:(testers[34]-1)),,,]

# Plot images to check that everything is ok
plot(train_x_data_final[1,1:img_height_exp,1:img_width_exp,3:1] %>% as.cimg())
plot(train_y_data_final[1,1:img_height_exp,1:img_width_exp,8] %>% as.cimg())
plot(train_y_data_final[1,1:img_height_exp,1:img_width_exp,11] %>% as.cimg())

# ---------------Export to npz----------------------------------------------------
# Use numpy to save everything as a npz file
np <- import("numpy")

np$savez(paste0("LULC_",input_type,"_Augment9perim_samples",".npz"), 
         x_train = train_x_data_final, 
         y_train = train_y_data_final, 
         x_test = test_x_data_final, 
         y_test = test_y_data_final)

