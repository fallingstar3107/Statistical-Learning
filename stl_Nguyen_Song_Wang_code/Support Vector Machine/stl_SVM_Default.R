library(MASS)
library(ISLR)
library(e1071)
library(penalizedSVM)
library(sparseSVM)
# read in the "Clients default" dataset in R
data = read.csv("default+of+credit+card+clients.csv", skip=1,header = TRUE)
#look at the dataset
str(data)
#create a new column y which sets "default" as 1 and "not default" as -1
data$y=ifelse(data$default.payment.next.month==0,-1,1)
#convert integer to categorical features
#data$SEX=as.factor(data$SEX)
#data$EDUCATION=as.factor(data$EDUCATION)
#data$MARRIAGE=as.factor(data$MARRIAGE)
#See how many people default next week
table(data$y)
barplot(prop.table(table(data$y)),
        col=rainbow(2),
        ylim=c(0,0.8),
        ylab="Proportion",
        xlab="default next month",
        cex.names = 1.5)
#from the barplot we found the data is imbalanced

#Data prep
data=as.matrix(data)
dimnames(data)=NULL
set.seed(1)
#set 0.7 of training and 0.3 of testing
ind=sample(2,nrow(data),replace = T,prob=c(0.7,0.3))
#set training_X,testing_X,training_Y and testing_Y
training=data[ind==1, 2:24]
testing=data[ind==2, 2:24]
trainingTarget=data[ind==1, 26]
testingTarget=data[ind==2, 26]

#Apply SVM with l2 norm on clients training data
datTrain=data.frame(x=training,
                    y=as.factor(trainingTarget))
datTest=data.frame(x=testing,
                   y=as.factor(testingTarget))
out=svm(y~.,data=datTrain,kernel="radial",
        gamma=20, cost=100)
summary(out)

#Confusion matrix on training data
table(out$fitted,datTrain$y)
#Confusion matrix on testing data
ypred=predict(out,datTest)
tab=table(predict=ypred,truth=datTest$y)
tab
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC
#use tune()to decide best gamma and cost
set.seed(1)
tune.out=tune(svm, y~., data=datTrain, kernel="radial", 
              ranges=list(cost=c(0.1,1,10,100,1000),
                          gamma=c(0.01,0.1,1,10,100)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)
table(bestmod$fitted,datTrain$y)
#Confusion matrix on testing data
ypred=predict(bestmod,datTest)
tab=table(predict=ypred,truth=datTest$y)
tab
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC
#try linear kernel
tune.out=tune(svm, y~., data=datTrain, kernel="linear", 
              ranges=list(cost=c(0.01,0.1,1,10)))

summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)
table(bestmod$fitted,datTrain$y)

#confusion matrix to see how many instances predicted wrong
datTest=data.frame(x=testing,y=as.factor(testingTarget))
ypred=predict(bestmod,datTest)
tab=table(predict=ypred,truth=datTest$y)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC

#computation time of svm with l1 norm
system.time(svm(y~.,data=datTrain,kernel="radial",
                gamma=0.01, cost=10))
system.time(svm(y~.,data=datTrain,kernel="linear",
                cost=1))

#lambda_range=c(seq(100,0.01,length.out=2))
#Apply SVM with l2 norm on clients training data
lambda_range=c(0.1,1,10,100)
norm1.fix=svmfs(x=data[ind==1, 2:24], y=data[ind==1,26], fs.method="1norm",
                cross.outer= 0, grid.search = "discrete",
                lambda1.set=lambda_range, calc.class.weights=TRUE, class_weights=list("-1":1, "1":3),
                parms.coding = "none", show="none",
                maxIter = 700, inner.val.method = "cv", cross.inner= 5,
                seed=1, verbose=FALSE )

str(norm1.fix)
norm1.fix$model
norm1.fix$model$lambda1
norm1.fix$model$xind
norm1.fix$model$cv.info
norm1.fix$model$fit.info

#ACC and ERR on testing data
test.error.1norm=predict(norm1.fix, 
                         newdata=testing,
                         newdata.labels=testingTarget)
print(test.error.1norm$error)
print(test.error.1norm$tab)


#Apply elastic Net approach L1+L2 norm
fit=cv.sparseSVM(X=training, y= trainingTarget, alpha = 0.5, gamma = 0.1, nlambda=100,
                 lambda.min =0.01,nfolds = 5,
                 preprocess = c("standardize", "rescale", "none"),
                 max.iter = 1000, eps = 1e-5)

#show min lambda
fit$lambda.min
#show number of variables selected
predict(fit, testing, lambda=0.01, type = c("nvars"), exact = FALSE) 
#show coefficients of each input 
predict(fit, testing, lambda=0.01, type = c("coefficients"), exact = FALSE)
#plot the cv graph
plot(fit)

#Use fit model to predict testing data with optimal lambda=0.01
ypred=predict(fit, testing, lambda = 0.01)
tab=table(predict=ypred,truth=datTest$y)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC
