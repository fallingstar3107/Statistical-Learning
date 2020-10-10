library(MASS)
library(ISLR)
library(e1071)
library(mlegp)
library(penalizedSVM)
library(sparseSVM)
library(ROCR)
#import Boston dataset
set.seed(1)
names(Boston)
attach(Boston)
#convert crim from number to label
catcrim=cut(Boston$crim,breaks=c(0,1,100),labels = c("0","1"))
dat=Boston
dat$catcrim=catcrim
#drop out original crim column in dataset
df = subset(dat, select = -c(crim) )
str(df)

# set 70% instances as training
train=sample(506,354)
#Apply l2 norm with SVM
svmfit=svm(catcrim~., data=df[train,], kernel="radial",  gamma=20, cost=100)
summary(svmfit)

ypred=predict(svmfit,df[-train,])
table(predict=ypred,truth=df[-train,]$catcrim)


#use tune()to decide best gamma and cost
set.seed(1)
tune.out=tune(svm, catcrim~., data=df[train,], kernel="radial", 
              ranges=list(cost=c(0.1,1,10,100,1000,10000),
                          gamma=c(0.001,0.01,0.1,1)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)
table(svmfit$fitted,df[train,]$catcrim)
#confusion matrix to see how many instances predicted wrong
ypred=predict(bestmod,dat[-train,])
tab= table(predict=ypred,truth=df[-train,]$catcrim)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC

#Misclassification Rate
p=predict(bestmod,df)
tab=table(p,catcrim)
correct_classRate=sum(diag(tab))/sum(tab)
miss_classRate=1-correct_classRate
miss_classRate

#ROC Curves of Training Data
library(ROCR)
svmfit.opt=svm(catcrim~., data=df[train,], kernel="radial",gamma=0.01, cost=10,decision.values=TRUE)
fitted=attributes(predict(svmfit.opt,df[train,],decision.values=TRUE))$decision.values
par(mfrow=c(1,2))

predob=prediction(fitted,df[train,"catcrim"])
perf=performance(predob,"tpr","fpr")
plot(perf,main="Training Data")

svmfit.flex=svm(catcrim~., data=df[train,], kernel="radial",gamma=0.01, cost=1000,decision.values=TRUE)
fitted=attributes(predict(svmfit.flex,df[train,],decision.values=TRUE))$decision.values

predob=prediction(fitted,df[train,"catcrim"])
perf=performance(predob,"tpr","fpr")
plot(perf,col="red", add=T, main="Training Data")

#ROC Curves of Testing Data
fitted=attributes(predict(svmfit.opt,df[-train,],decision.values=TRUE))$decision.values
predob=prediction(fitted,df[-train,"catcrim"])
perf=performance(predob,"tpr","fpr")
plot(perf,main="Testing Data")

fitted=attributes(predict(svmfit.flex,df[-train,],decision.values=TRUE))$decision.values
predob=prediction(fitted,df[-train,"catcrim"])
perf=performance(predob,"tpr","fpr")
plot(perf,col="red", add=T,main="Testing Data")

# Construct dataset
colA <- Boston$zn
colB <- Boston$indus
colC <- Boston$chas
colD <- Boston$nox
colE <- Boston$rm
colF <- Boston$age
colG <- Boston$dis
colH <- Boston$rad
colI <- Boston$tax
colJ <- Boston$ptratio
colK <- Boston$black
colL <- Boston$lstat
colM <- Boston$medv
x <- cbind(colA, colB,colC,colD,colE,colF,colG,colH,colI,colJ,colK,colL,colM)
y=df$catcrim
y=as.numeric(y)
datfram=data.frame(x=x,y=y)
#convert label to -1 and +1
datfram[datfram$y==1,]$y=-1
datfram[datfram$y==2,]$y=1

trainL1=sample(506,354)
#Apply L1 norm on svm
lambda_range=c(seq(100,0.01,length.out=10))
norm1.fix=svmfs(x=x[trainL1,], y=datfram[trainL1,]$y, fs.method="1norm",
                cross.outer= 0, grid.search = "discrete",
                lambda1.set=lambda_range,calc.class.weights=TRUE, class_weights=list("-1":1, "1":1.91),
                parms.coding = "none", show="none",
                maxIter = 700, inner.val.method = "cv", cross.inner= 5,
                seed=1, verbose=FALSE )
                
str(norm1.fix)
norm1.fix$model
norm1.fix$model$lambda1
norm1.fix$model$xind
norm1.fix$model$cv.info
norm1.fix$model$fit.info

#predict testing data
ypred=predict(norm1.fix,datfram[-trainL1,])
tab=table(predict=ypred$pred.class,truth=datfram[-trainL1,]$y)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC

#Apply Elastic net norm on Boston data
fit=cv.sparseSVM(X=x[trainL1,], y=datfram[trainL1,]$y, alpha = 0.5, gamma = 0.1, nlambda=100,
                 lambda.min =0.01,
                 preprocess = c("standardize", "rescale", "none"),
                 max.iter = 1000, eps = 1e-5)
#show min lambda
fit$lambda.min
#show number of variables selected
predict(fit, x[-train,], lambda=0.38, type = c("nvars"), exact = FALSE) 
#show coefficients of each input 
predict(fit, x[-train,], lambda=0.38, type = c("coefficients"), exact = FALSE)
#plot the cv graph
plot(fit)

#Use fit model to predict testing data with optimal lambda=0.01
ypred=predict(fit, x[-train,], lambda = 0.38)
tab=table(predict=ypred,truth=datfram[-trainL1,]$y)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC

#try other alpha = 0.7 value
fit=cv.sparseSVM(X=x[trainL1,], y=datfram[trainL1,]$y, alpha = 0.7, gamma = 0.1, nlambda=100,
                 lambda.min =0.01,
                 preprocess = c("standardize", "rescale", "none"),
                 max.iter = 1000, eps = 1e-5)
#show min lambda
fit$lambda.min
#show number of variables selected
predict(fit, x[-train,], lambda=0.38, type = c("nvars"), exact = FALSE) 
#show coefficients of each input 
predict(fit, x[-train,], lambda=0.38, type = c("coefficients"), exact = FALSE)
#plot the cv graph
plot(fit)

#Use fit model to predict testing data with optimal lambda=0.01
ypred=predict(fit, x[-train,], lambda = 0.38)
tab=table(predict=ypred,truth=datfram[-trainL1,]$y)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC

#try other alpha = 0.3 value
fit=cv.sparseSVM(X=x[trainL1,], y=datfram[trainL1,]$y, alpha = 0.3, gamma = 0.1, nlambda=100,
                 lambda.min =0.01,
                 preprocess = c("standardize", "rescale", "none"),
                 max.iter = 1000, eps = 1e-5)
#show min lambda
fit$lambda.min
#show number of variables selected
predict(fit, x[-train,], lambda=0.38, type = c("nvars"), exact = FALSE) 
#show coefficients of each input 
predict(fit, x[-train,], lambda=0.38, type = c("coefficients"), exact = FALSE)
#plot the cv graph
plot(fit)

#Use fit model to predict testing data with optimal lambda=0.01
ypred=predict(fit, x[-train,], lambda = 0.38)
tab=table(predict=ypred,truth=datfram[-trainL1,]$y)
#show ACC and ERR
ACC=sum(diag(tab))/sum(tab)
ERR=1-ACC
ERR
ACC



