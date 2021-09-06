library(raster)
library(reticulate)
library(sf)

# Number of squared samples (windows of x by x pixels) for which you have training data
num_imagenes <- 33

# Input info for U-net
input_type <- "OpticRadar"

# Use numbers to name training data of image samples (reflectance and backscatter values) 
# and image masks (land use land cover [LULC] categories)
nombres_mask <- seq(1,num_imagenes,1)

# Number of LULC classes
n_classes <- 12

# Number of bands of the input image
channels <- 6
# radar
# channels <- 2
# optic
# channels <- 4


# Size of windows in pixels (training data)
img_width <- 256
img_height <- 256
# Size of windows in pixels (augumented training data)
img_width_exp <- 128
img_height_exp <- 128


# Number of 128pix-squares that are going to be obtained per 256 pix-squares
# In this case we are going to use 25 (5 x 5 128 x 128 pix) using a 32 pix offset
# This is cropping the image as 1-129, 1-129; 33-161, 1-129; 65-193, 1-129; 97-225, 1-129; 129-256, 1-129 
num_squares <- 3^2
probs_4crops <- 1 / (sqrt(num_squares)-1) 
# Number of mirrored images per 128pix-image
num_mirrors <- 2
#----------------------First graticule-------------------------------------------------
# Read Training Info
# Locations of image files
optic_names <- "4BSentinel-2img.tif"

# Esta versión ya está leyendo los datos originales de backscatter en gamma0
radar_names <- "2BSentinel-1img.tif"

# ROI shp
area <- st_read("roi.shp")
plot(area)

# Read images as stacks
optic <- stack(optic_names)
radar <- stack(radar_names)

# Hay que cortar la radar con optic porque no da la misma extension ??? por qué??
radar <- crop(radar, optic)

# Input data that is going to be pre processed to further feed the U-net

# Radar Optic
# Features: Stack optic 4 bands and 2 radar bands
optic <- stack(optic,radar[[1:2]])

rm(radar)
# Optic
# optic <- optic
# Radar
# optic <- radar[[1:2]]

# Crop
optic <- crop(optic, area)

dim(optic)


# Create the sequence of splits
splits <- c(floor(dim(optic)[1] / 128),floor(dim(optic)[2] / 128))

# Create sequence of splits in x axis and y axis
x_splits <- seq(1, (floor(dim(optic)[1] / 128) * 128), 128)
y_splits <- seq(1, (floor(dim(optic)[2] / 128) * 128), 128)

# Create empty arrays to fill it with the previous info
pred_x_data<-array(0,
                    dim = c(length(x_splits) * length(y_splits),
                            img_width_exp,
                            img_width_exp,
                            channels))

# Convert image as array
optic <- as.array(optic)

# Fill the arrays
aux <- 1
for(i in 1:(length(x_splits)-1)){
  for(j in 1:(length(y_splits)-1)){
    pred_x_data[aux,1:128,1:128,1:channels] <- optic[x_splits[i]:(x_splits[i]+127),
                                                     y_splits[j]:(y_splits[j]+127),
                                                     1:channels]
    aux <- aux+1
  }    
}

# Plot an example
plot(as.raster(pred_x_data[2700,1:128,1:128,1:3]/2000))

# once we've got the arrays we need to normalize the data according to mean and sd values
bands_mean <- read.csv(paste0("Mean_x",input_type,".csv"))[,1]
bands_sd <- read.csv(paste0("sd_x",input_type,".csv"))[,1]

# Normalize data (only features data)
for(i in 1:dim(pred_x_data)[1]){
  for(j in 1:dim(pred_x_data)[4]){
    pred_x_data[i,,,j] <- (pred_x_data[i,,,j] - bands_mean[j]) / bands_sd[j]
  }
}

# See example
plot(as.raster((pred_x_data[2700,1:128,1:128,1:3]+2)/4))

# This option was prefered as a single file can contain both training and test data
np <- import("numpy")

np$savez(paste0("LULC_fullImg_a",input_type,".npz"), 
         x_test = pred_x_data)

# -----------------------------Second graticule-----------------------------------------
# No need to reload and crop optic and radar.

dim(optic)

# one less than the original split because this starts with an offset of 64, son 1 tile less
splits <- c(floor((dim(optic)[1]-65) / 128),floor((dim(optic)[2]-65) / 128))

x_splits <- seq(65, (floor((dim(optic)[1]-65) / 128) * 128), 128)
y_splits <- seq(65, (floor((dim(optic)[2]-65) / 128) * 128), 128)

# Create empty arrays to fill it with the previous info
pred_x_data<-array(0,
                   dim = c(length(x_splits) * length(y_splits),
                           img_width_exp,
                           img_width_exp,
                           channels))

# Fill the arrays
aux <- 1
for(i in 1:(length(x_splits)-1)){
  for(j in 1:(length(y_splits)-1)){
    pred_x_data[aux,1:128,1:128,1:channels] <- optic[x_splits[i]:(x_splits[i]+127),
                                                     y_splits[j]:(y_splits[j]+127),
                                                     1:channels]
    aux <- aux+1
  }    
}

plot(as.raster(pred_x_data[2800,1:128,1:128,1:3]/2000))

# once we've got the arrays we need to normalize the data according to mean and sd values
# This is already done so we'll just comment it
# bands_mean <- read.csv(paste0("Mean_x",input_type,".csv"))[,1]
# bands_sd <- read.csv(paste0("sd_x",input_type,".csv"))[,1]

# Normalize data (only features data)
for(i in 1:dim(pred_x_data)[1]){
  for(j in 1:dim(pred_x_data)[4]){
    pred_x_data[i,,,j] <- (pred_x_data[i,,,j] - bands_mean[j]) / bands_sd[j]
  }
}

plot(as.raster((pred_x_data[2800,1:128,1:128,1:3]+2)/4))

# Other option: saving files as numpy zip files
# This option was prefered as a single file can contain both training and test data
np <- import("numpy")

np$savez(paste0("LULCfullImg_b",input_type,".npz"), 
         x_test = pred_x_data)
