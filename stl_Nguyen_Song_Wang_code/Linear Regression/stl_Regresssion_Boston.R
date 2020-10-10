library(MASS) # Data source
library(knitr) # kable light weight table generator 
library(car)
library(glmnet)
library(caret)
data(Boston) # load data

lm.crim <- lm(crim ~ ., data = Boston[train,])
lm.crim.summary <- summary(lm.crim)
lm.crim.summary

lm.predict  = predict(lm.crim, newdata =Boston[test,])
lm.MSE = mean((lm.predict-y.test)^2)
lm.MSE


### Ridge Regression

sum(is.na(Boston$crim)) #check NA

# Create x matrix
x = model.matrix(crim~.,Boston)[,-1]
y = Boston$crim # Create response vector y

set.seed(1)

# Get Training and Testing Data Set
train = sample(1:nrow(x), 3*nrow(x)/5) # ~300 obs for 10-fold CV, each fold has ~30 obs.
test = (-train)
y.test = y[test]

# Fit ridge regression on the training set
grid = 10^seq(10, -2, length=100)
ridge.mod = glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)

# Perform k-fold cross validation on all the data to determine best lambda value
ridge.cv.out = cv.glmnet(x[train,], y[train], alpha=0) # this does 10-fold cv by default, appropriate with 500 obs
ridge.bestlam = ridge.cv.out$lambda.min

# MSE associated with value of lambda that results in smallest cv-error
ridge.pred = predict(ridge.mod, s=ridge.bestlam, newx=x[test,])
ridge.MSE = mean((ridge.pred - y.test)^2)

# Now refit ridge regression on full data set, using the value of lambda
# selected by cross-validation
ridge.out = glmnet(x, y, alpha=0, lambda=grid)
ridge.coef = predict(ridge.out, type="coefficients", s=ridge.bestlam)[1:13,]

# Save off Results
ridge.results = data.frame(as.list(ridge.coef))
ridge.results$'Test MSE' = ridge.MSE
ridge.results$BestLambda = ridge.bestlam
row.names(ridge.results) = 'RidgeRegression'

# Rearrange variables 
ridge.results = ridge.results[,c(which(colnames(ridge.results)=="BestLambda"), which(colnames(ridge.results)!="BestLambda"))]
ridge.results = ridge.results[,c(which(colnames(ridge.results)=="Test MSE"), which(colnames(ridge.results)!="Test MSE"))]
ridge.results = t(ridge.results) # transpose for easier table display

# display results in table
kable(ridge.results, format = 'pipe')


###Lasso

set.seed(1)

# Fit lasso on the training set (same one used as ridge)
grid = 10^seq(10, -2, length=100) # using same grid as before (optional: could use new grid)
lasso.mod = glmnet(x[train,], y[train], alpha=1, lambda=grid, thresh=1e-12)

# Create lasso model, perform k-fold CV, and get best lambda and lowest MSE
lasso.cv.out = cv.glmnet(x[train,], y[train], alpha=1)
lasso.bestlam = lasso.cv.out$lambda.min

lasso.pred = predict(lasso.mod, s=lasso.bestlam, newx=x[test,])
lasso.MSE = mean((lasso.pred - y.test)^2) # test set MSE

# Now refit lasso on full data set, using the value of lambda
# selected by cross-validation
lasso.out = glmnet(x, y, alpha=1, lambda=grid)
lasso.coef = predict(lasso.out, type="coefficients", s=lasso.bestlam)[1:13,]

# Get coefficients and save off results
lasso.coef = data.frame(as.list(lasso.coef))
lasso.results = data.frame(as.list(lasso.coef))
lasso.results$'Test MSE' = lasso.MSE
lasso.results$BestLambda = lasso.bestlam
row.names(lasso.results) = 'Lasso'

# Rearrange variables
lasso.results = lasso.results[,c(which(colnames(lasso.results)=="BestLambda"), which(colnames(lasso.results)!="BestLambda"))]
lasso.results = lasso.results[,c(which(colnames(lasso.results)=="Test MSE"), which(colnames(lasso.results)!="Test MSE"))]
lasso.results = t(lasso.results) # transpose for easier table display

# display results in table
kable(lasso.results, format = 'pipe')


###elastic net

#set training control
train_cont = trainControl(method = "repeatedcv", 
                          number=10, 
                          repeats=5, 
                          search="random", 
                          verboseIter=TRUE)

#train the model
elastic_reg = train(crim~., data=Boston,
                    method="glmnet",
                    preProcess=c("center", "scale"),
                    tuneLength=10,
                    trControl=train_cont)
elastic_reg$bestTune

set.seed(1)

# Fit elastic on the training set (same one used as ridge)
grid = 10^seq(10, -2, length=100) # using same grid as before (optional: could use new grid)
elastic.mod = glmnet(x[train,], y[train], 
                   alpha=elastic_reg$bestTune$alpha, 
                   lambda=elastic_reg$bestTune$lambda, thresh=1e-12)

# Create elastic model, perform k-fold CV, and get best lambda and lowest MSE
elastic.cv.out = cv.glmnet(x[train,], y[train], alpha=elastic_reg$bestTune$alpha)
elastic.pred = predict(elastic.mod, newx=x[test,])
elastic.MSE = mean((elastic.pred - y.test)^2) # test set MSE

# Now refit elastic on full data set, using the value of lambda
# selected by cross-validation
elastic.out = glmnet(x, y, alpha=elastic_reg$bestTune$alpha, 
                   lambda=elastic_reg$bestTune$lambda)
elastic.coef = predict(elastic.out, type="coefficients")[1:13,]

# Get coefficients and save off results
elastic.coef = data.frame(as.list(elastic.coef))
elastic.results = data.frame(as.list(elastic.coef))
elastic.results$'Test MSE' = elastic.MSE
row.names(elastic.results) = 'ElasticNet'

# Rearrange variables
#elastic.results = elastic.results[,c(which(colnames(elastic.results)=="BestLambda"), which(colnames(elastic.results)!="BestLambda"))]
elastic.results$BestLambda = elastic_reg$bestTune$lambda
elastic.results = elastic.results[,c(which(colnames(elastic.results)=="Test MSE"), which(colnames(elastic.results)!="Test MSE"))]
elastic.results = t(elastic.results) # transpose for easier table display

# display results in table
kable(elastic.results, format = 'pipe')


# Merge/Combine Results
rownameOrder = rownames(ridge.results) # To save order of row names
comparisonResults = merge(cbind(ridge.results, lasso.results), elastic.results, by = 0, all = TRUE)
currRowNames = comparisonResults$Row.names
comparisonResults = comparisonResults[-1] #Remove Row.Names column
rownames(comparisonResults) = currRowNames # Add real row names
comparisonResults=comparisonResults[rownameOrder,] # Change order of rows
#comparisonResults[comparisonResults == 0] = NA # Change 0 to NA (shows up blank in table)
# display results in table
kable(comparisonResults, format = 'pipe')

