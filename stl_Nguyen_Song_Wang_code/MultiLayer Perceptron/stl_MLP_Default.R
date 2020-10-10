#------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION
#------------------------------------------------------------------------
install.packages(tensorflow)
install.packages(httr)
install.packages(RCurl)
install.packages(readxl)
install.packages(remotes)
remotes::install_github("rstudio/tensorflow")
install.packages(ggplot2)
install.packages(matrixStats)
install.packages(RSNNS)

library(tensorflow)
install_tensorflow(version = "2.0.0")
library(keras)
library(httr)
library(RCurl)
library(readxl)
library(matrixStats)
library(RSNNS)
library(caret)
library(tidyverse)
library(pROC)
#------------------------------------------------------------------------
## 2. DATA PREPROCESSING
#------------------------------------------------------------------------
# Get data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
GET(url, write_disk("default.xls", overwrite=TRUE))
dataset <- read_xls('default.xls', sheet = 1, skip = 1)
# Drop the index column and rename the target variable column 
dataset$ID <- NULL
colnames(dataset)[24] <- "default"
# Remove missing values
dataset <- na.omit(dataset)
with(dataset, sum(is.na(default)))

#------------------------------------------------------------------------
## 3. DATA SPLIT: TRAIN DATASET AND TEST DATASET 
#------------------------------------------------------------------------
# Random sampling, create training (80%) and test set (20%)
set.seed(1)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
X_train = as.matrix(dataset[index , 1:23])
Y_train = as.matrix(dataset[index , 24])
X_test = as.matrix(dataset[-index , 1:23])
Y_test = as.matrix(dataset[-index , 24])

#------------------------------------------------------------------------
## 4. DATA SCALING 
#------------------------------------------------------------------------
# Normalize data, range from 0 to 1
scaled_X_train = normalizeData(X_train, type = "0_1")
scaled_X_test = normalizeData(X_test, type = "0_1")
scaled_Y_train = normalizeData(Y_train, type = "0_1")
scaled_Y_test = normalizeData(Y_test, type = "0_1")

#------------------------------------------------------------------------
## 5. CONSTRUCT MODELS
#------------------------------------------------------------------------
set.seed(1)
mlp <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0)) %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l2 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l2(l = 0.0001)) %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l2(l = 0.0001))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))
  
set.seed(1)
mlp_l1 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1(l = 0.0001)) %>% 
  #layer_dropout(rate = 0.5)%>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1(l = 0.0001))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l1_l2 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001 )) %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
opt<-optimizer_adam( lr= 0.00001 , decay = 0, clipnorm = 1 )
mlp %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))
mlp_l2 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))
mlp_l1 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))
mlp_l1_l2 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))

mlp %>% summary()
mlp_l2 %>% summary()
mlp_l1 %>% summary()
mlp_l1_l2 %>% summary()

# TRAIN/FIT HISTORIES
# K-FOLD CROSS VALIDATION -------------------------------------------------
k <- 5
indices <- sample(1:nrow(scaled_X_train))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 

acc_histories_mlp <- NULL
acc_histories_mlp_l2 <- NULL
acc_histories_mlp_l1 <- NULL
acc_histories_mlp_l1_l2 <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # TRAINING/VALIDATION DATA PARTITION ----------------------------------
  
  # Prepare the validation data: data from partition # k
  indices_val <- which(folds == i, arr.ind = TRUE) 
  scaled_X_val <- scaled_X_train[indices_val,]
  scaled_Y_val <- scaled_Y_train[indices_val]
  
  # Prepare the training data: data from all other partitions
  partial_scaled_X_train <- scaled_X_train[-indices_val,]
  partial_scaled_Y_train <- scaled_Y_train[-indices_val]
  
  #Train the NN
  set.seed(1)
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 80 == 0) cat("\n")
      cat(".")
    }
  )  
  
  # set class weight for the imbalanced dataset
  set.seed(1)
  total = nrow(dataset)
  neg = binCounts(dataset$default, bx =c(0, 1))
  pos = total - neg
  
  #Evaluate the models
  set.seed(1)
  history_mlp <- mlp %>% 
    fit(partial_scaled_X_train,
        partial_scaled_Y_train,
        validation_data = list(scaled_X_val, scaled_Y_val),
        epochs = 100,
        verbose = 0, 
        class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)), 
        callbacks = list(print_dot_callback))
  plot(history_mlp, metrics = "accuracy", smooth = FALSE)
  
  set.seed(1)
  history_mlp_l2 <- mlp_l2 %>% 
    fit(partial_scaled_X_train,
        partial_scaled_Y_train,
        validation_data = list(scaled_X_val, scaled_Y_val),
        epochs = 100,
        verbose = 0, 
        class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)), 
        callbacks = list(print_dot_callback))
  plot(history_mlp_l2, metrics = "accuracy", smooth = FALSE) 
  
  set.seed(1)
  history_mlp_l1 <- mlp_l1 %>% 
    fit(partial_scaled_X_train,
        partial_scaled_Y_train,
        validation_data = list(scaled_X_val, scaled_Y_val),
        epochs = 100,
        verbose = 0,   
        class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)), 
        callbacks = list(print_dot_callback))
  plot(history_mlp_l1, metrics = "accuracy", smooth = FALSE) 
  
  set.seed(1)
  history_mlp_l1_l2 <- mlp_l1_l2 %>% 
    fit(partial_scaled_X_train,
        partial_scaled_Y_train,
        validation_data = list(scaled_X_val, scaled_Y_val),
        epochs = 100,
        verbose = 0, 
        class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)), 
        callbacks = list(print_dot_callback))
  plot(history_mlp_l1_l2, metrics = "accuracy", smooth = FALSE) 
  
  # LOAD TRAIN/FIT HISTORIES -----------------------------------------------------
  
  add_acc_history_mlp <- mlp %>%  evaluate(scaled_X_val, scaled_Y_val, verbose = 0)
  acc_histories_mlp <- c(acc_histories_mlp, 
                         add_acc_history_mlp["accuracy"])
  
  add_acc_history_mlp_l2 <- mlp_l2 %>%  evaluate(scaled_X_val, scaled_Y_val, verbose = 0)
  acc_histories_mlp_l2 <- c(acc_histories_mlp_l2, 
                            add_acc_history_mlp_l2["accuracy"])
  
  add_acc_history_mlp_l1 <- mlp_l1 %>%  evaluate(scaled_X_val, scaled_Y_val, verbose = 0)
  acc_histories_mlp_l1 <- c(acc_histories_mlp_l1, 
                            add_acc_history_mlp_l1["accuracy"])
  
  add_acc_history_mlp_l1_l2 <- mlp_l1_l2 %>%  evaluate(scaled_X_val, scaled_Y_val, verbose = 0)
  acc_histories_mlp_l1_l2 <- c(acc_histories_mlp_l1_l2, 
                               add_acc_history_mlp_l1_l2["accuracy"])
  
}

# Compute the average train/val MSE for all folds of 4 models

paste0("Average accuracy of MLP on validation set: ", sprintf("%.2f", mean(acc_histories_mlp)))
paste0("Average accuracy of MLP (L2 regularisation) on validation set: ", sprintf("%.2f", mean(acc_histories_mlp_l2)))
paste0("Average accuracy of MLP (L1 regularisation) on validation set: ", sprintf("%.2f", mean(acc_histories_mlp_l1)))
paste0("Average accuracy of MLP (L1 and L2 regularisation) on validation set: ", sprintf("%.2f", mean(acc_histories_mlp_l1_l2)))

#------------------------------------------------------------------------
## 6. TEST NN MODEL
#------------------------------------------------------------------------
set.seed(1)
pred_mlp <- predict_classes(mlp, scaled_X_test)
table(Y_test, pred_mlp)
mean(Y_test == pred_mlp)

set.seed(1)
pred_mlp_l2 <- predict_classes(mlp_l2, scaled_X_test)
table(Y_test, pred_mlp_l2)
mean(Y_test == pred_mlp_l2)

set.seed(1)
pred_mlp_l1 <- predict_classes(mlp_l1, scaled_X_test)
table(Y_test, pred_mlp_l1)
mean(Y_test == pred_mlp_l1)

set.seed(1)
pred_mlp_l1_l2 <- predict_classes(mlp_l1_l2, scaled_X_test)
table(Y_test, pred_mlp_l1_l2)
mean(Y_test == pred_mlp_l1_l2)

df <- data.frame("Actual" = Y_test, "Baseline" = pred_mlp, "L2" = pred_mlp_l2, "L1" = pred_mlp_l1, "L1 and L2" = pred_mlp_l1_l2)
head(df[1:5], 10)
summary(df)

# Calculate the AUC of MLP 
roc_obj <- roc(Y_test, pred_mlp)
auc(roc_obj)

# Calculate the AUC of MLP L2
roc_obj <- roc(Y_test, pred_mlp_l2)
auc(roc_obj)

# Calculate the AUC of MLP L1
roc_obj <- roc(Y_test, pred_mlp_l1)
auc(roc_obj)

# Calculate the AUC of MLP L1_L2
roc_obj <- roc(Y_test, pred_mlp_l1_l2)
auc(roc_obj)

