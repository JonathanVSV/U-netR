# Load libraries
library(keras)
library(tfruns)

# Set memory limit
memory.limit(size=56000)
memory.limit()

# Set type of imagery used
imagery <- "OpticRadar"

# Hyperparameter exploration
runs <- tuning_run(paste0("U-net_train",".R"),
                   runs_dir = paste0("hyperparam_tuning_8-32batch_",imagery),
                   # Make sample or not, 1 tests all combinations
                   sample = 1,
                   # Flags, i.e., values that are going to be explortes
                   flags = list(batch_size = c(8,16,32),
                                learn_rate = c(1e-4),
                                epochs = c(50),
                                dropout = seq(0.1,0.5,0.1), 
                                filters_firstlayer = c(32, 64),
                                num_layers = c(2,3,4)),
                   echo = F,
                   confirm = F) 



# Write csv containing the summary of the exploration
write.csv(runs, 
          paste0("Runs_hyperparam_tuning_8-32batch_",imagery,".csv"),
          row.names = F)
