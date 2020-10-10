library(knitr) # kable light weight table generator 
library(car)
library(glmnet)
library(caret)
library(ROCR)
library(ggplot2)

setwd(dir = "/Users/songpengwei/Desktop/stl_default")
getwd()
dataset <- read.csv("default.csv", header=T, skip = 1)

# Drop the index column and rename the target variable column 
dataset$ID <- NULL
colnames(dataset)[24] <- "default"
# Remove missing values
dataset <- na.omit(dataset)
with(dataset, sum(is.na(default)))
# Explore the data

dim(dataset)
#DATA SPLIT: TRAIN DATASET AND TEST DATASET 
# Random sampling, create training (80%) and test set (20%)
set.seed(80)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
train = dataset[index , 1:24]
test = dataset[-index , 1:24]


#fit the train model
glm.fit <- glm(default~ ., family = binomial, data = train)

summary(glm.fit)

#access the predictive ability of the model
glm.probs <- predict(glm.fit, test, type="response")


glm.pred=rep("0", 6000)
glm.pred[glm.probs>.5]="1"


accuracy = mean(glm.pred==test$default)
print(paste('Accuracy', accuracy))


#ROC curve and AUC

p <- predict(glm.fit, test, type="response")
pr <- prediction(p, test$default)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

####Regularization

x.train = as.matrix(train[,1:23]) # Create x matrix
y.train = as.factor(train[,24]) # Create response vector y
x.test = as.matrix(test[,1:23])
y.test = as.factor(test[,24])


###Ridge Regression

sum(is.na(dataset$default)) #check NA
set.seed(1)
# Perform k-fold cross validation on all the data to determine best lambda value
ridge.cv.out = cv.glmnet(x.train, y.train, alpha=0, type.measure="class", nfolds=10,family="binomial") # this does 10-fold cv by default, appropriate with 500 obs
ridge.bestlam = ridge.cv.out$lambda.min

# Fit ridge regression on the training set

ridge.mod = glmnet(x.train, y.train, alpha=0, family="binomial")

#access the predictive ability of the model

ridge.probs = predict(ridge.mod, x.test, s=ridge.bestlam,type = "class")

#model accuracy
ridge.pred = rep("0", 6000)
ridge.pred[ridge.probs>.5]="1"
ridge.accuracy = mean(ridge.pred==test$default)
print(paste('Ridge.Accuracy', ridge.accuracy))

#ROC curve and AUC

ridge.p <- predict(ridge.mod, x.test, s=ridge.bestlam, type="response")
ridge.pr <- prediction(ridge.p, test$default)
ridge.prf <- performance(ridge.pr, measure = "tpr", x.measure = "fpr")
plot(ridge.prf)

ridge.auc <- performance(ridge.pr, measure = "auc")
ridge.auc <- ridge.auc@y.values[[1]]
ridge.auc


###Lasso

set.seed(1)
# Perform k-fold cross validation on all the data to determine best lambda value
lasso.cv.out = cv.glmnet(x.train, y.train, alpha=1, type.measure="class", nfolds=10,family="binomial") # this does 10-fold cv by default, appropriate with 500 obs
lasso.bestlam = lasso.cv.out$lambda.min

# Fit ridge regression on the training set

lasso.mod = glmnet(x.train, y.train, alpha=1, family="binomial")

#access the predictive ability of the model

lasso.probs = predict(lasso.mod, x.test, s=lasso.bestlam, type = "class")

#model accuracy
lasso.pred = rep("0", 6000)
lasso.pred[lasso.probs>.5]="1"
lasso.accuracy = mean(lasso.pred==test$default)
print(paste('Lasso.Accuracy', lasso.accuracy))

#ROC curve and AUC

lasso.p <- predict(lasso.mod, x.test, s=lasso.bestlam, type="response")
lasso.pr <- prediction(lasso.p, test$default)
lasso.prf <- performance(lasso.pr, measure = "tpr", x.measure = "fpr")
plot(lasso.prf)

lasso.auc <- performance(lasso.pr, measure = "auc")
lasso.auc <- lasso.auc@y.values[[1]]
lasso.auc


###elastic net

#set training control
train_cont = trainControl(method = "repeatedcv", 
                          number=10, 
                          repeats=5, 
                          search="random", 
                          verboseIter=TRUE)

#train the model
elastic_reg = train(default~., data=dataset,
                    method="glmnet",
                    preProcess=c("center", "scale"),
                    tuneLength=10,
                    trControl=train_cont)
elastic_reg$bestTune

set.seed(1)

# Perform k-fold cross validation with the best lambda value
elastic.cv.out = cv.glmnet(x.train, y.train, alpha=elastic_reg$bestTune$alpha, type.measure="class", nfolds=10,family="binomial") # this does 10-fold cv by default, appropriate with 500 obs
elastic.bestlam = elastic_reg$bestTune$lambda

# Fit ridge regression on the training set

elastic.mod = glmnet(x.train, y.train, alpha=elastic_reg$bestTune$alpha, family="binomial")

#access the predictive ability of the model

elastic.probs = predict(elastic.mod, x.test, s=elastic.bestlam, type = "class")

#model accuracy
elastic.pred = rep("0", 6000)
elastic.pred[elastic.probs>.5]="1"
elastic.accuracy = mean(elastic.pred==test$default)
print(paste('Elastic.Accuracy', elastic.accuracy))

#ROC curve and AUC

elastic.p <- predict(elastic.mod, x.test, s=elastic.bestlam, type="response")
elastic.pr <- prediction(elastic.p, test$default)
elastic.prf <- performance(elastic.pr, measure = "tpr", x.measure = "fpr")
plot(elastic.prf)

elastic.auc <- performance(elastic.pr, measure = "auc")
elastic.auc <- elastic.auc@y.values[[1]]
elastic.auc
