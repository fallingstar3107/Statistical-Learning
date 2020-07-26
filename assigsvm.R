library(MASS)
library(ISLR)
library(e1071)

set.seed(1)
names(Boston)
attach(Boston)

catcrim=cut(Boston$crim,breaks=c(0,1,100),labels = c("0","1"))
dat=Boston
dat$catcrim=catcrim
df = subset(dat, select = -c(crim) )
str(df)

train=sample(506,354)
svmfit=svm(catcrim~., data=df[train,], kernel="radial",  gamma=20, cost=100)
plot(svmfit, dat[train,])
summary(svmfit)

ypred=predict(svmfit,dat[-train,])
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
table(predict=ypred,truth=df[-train,]$catcrim)

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
