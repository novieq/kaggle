###################################################
## Step 0: Function Declaration
###################################################
rm(list = ls(all = TRUE))
gc()
#Missing Value Treatment
f=function(x){
    x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
    x[is.na(x)] =mean(x, na.rm=TRUE) #convert the item with NA to median value from the column
    x #display the column
  
}

#Cross Validation Error 
logarithmicLogFun <- function(actual, prediction) {
  epsilon <- .000000000000001
  yhat <- pmin(pmax(prediction, rep(epsilon)), 1-rep(epsilon))
  logloss <- -mean(actual*log(yhat)
                   + (1-actual)*log(1 - yhat))
  return(logloss)
}

###################################################
## Step 1: Data Preparation
###################################################


library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
# Load Training data
print(Sys.time())
initial <- read.csv("/Users/sayghosh/code/kaggle/test/train_ctr.csv", nrows = 100)
classes <- sapply(initial, class)
classes
train = read.csv("/Users/sayghosh/code/kaggle/test/train_ctr.csv", colClasses = classes, nrows=100001, comment.char="")
print(Sys.time())

str(train)
summary(train)
train$Id<-NULL
train$Label<-as.factor(train$Label)

###################################################
## Step 2: Model Building
###################################################

# 1. Name : Using all the integer variables, NA replaced by median in test, NA left in train
#    Error :  0.5771638 
#    Training Size : 100001
mylogit <- glm(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13 , data = train, family = "binomial", na.action = "na.omit")
summary(mylogit)

# 2. Name : Using all the integer variables, NA replaced by median in test, NA replaced in train
#    Error :  0.5404027 , 0.5403759, 0.5403125, 0.5385134
#    Training Size : 100001, 200001, 10000001 , mean,
# By increasing size of the training sample, the CV error does not reduce, let us work on adding more features
train_=data.frame(apply(train,2,f))
train_[,15:40] <- train[15:40]
train_[,1] <- train[,1]
str(train)
str(train_)
summary(train)
summary(train_)
rm(train)

mylogit <- glm(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13 +C1 +C2 +C3 + C4 +C5 + C6 + C7 + C8 +C9 + C10 + C11 + C12 + C13 + C14 + C15 + C16 + C17 + C18 + C19 + C20 + C21 + C22 + C23 + C24 + C25 + C26, data = train_, family = "binomial", na.action = "na.omit")
summary(mylogit)


# 3. Name : Using only factor variable
#    Error :  0.5404027 , 0.5403759, 0.5403125, 0.5385134
#    Variable : C1, C2, C3, C4, C5, C6
# By increasing size of the training sample, the CV error does not reduce, let us work on adding more features
train_=data.frame(apply(train,2,f))
train_[,15:40] <- train[15:40]
train_[,1] <- train[,1]
str(train)
str(train_)
summary(train)
summary(train_)

mylogit <- glm(Label ~  C2, data = train_, family = "binomial", na.action = "na.omit")
summary(mylogit)

###################################################
## Step 3: Model Cross Validation 
###################################################

print(Sys.time())
test_missing_cv <- read.csv("/Users/sayghosh/code/kaggle/test/test_cv_ctr.csv", colClasses = classes, nrows=1000001)
print(Sys.time())

test_missing_cv$Id<-NULL
test_cv=data.frame(apply(test_missing_cv,2,f))
test_cv[,15:40] <- test_missing_cv[15:40]
test_cv[,1] <- test_missing_cv[,1]
str(test_missing_cv)
test_missing_cv$X <-NULL
str(test_cv)
test_cv$Label<-as.factor(test_cv$Label)
rm(test_missing_cv)
summary(test_cv)
test_cv$Prediction <- predict(mylogit, newdata = test_cv, type="response", se.fit=FALSE, na.action="na.omit")
cv_error <- logarithmicLogFun( as.numeric(test_cv$Label), test_cv$Prediction)
cv_error
help(predict)
###################################################
## Step 4: Predicting on test set
###################################################

initial <- read.csv("/Users/sayghosh/code/kaggle/test/test_ctr.csv", nrows = 100)
classes <- sapply(initial, class)
classes
print(Sys.time())
test <- read.csv("/Users/sayghosh/code/kaggle/test/test_ctr.csv", colClasses = classes, nrows=6042136)
print(Sys.time())

#Remove missing values from test data
test_missing=data.frame(apply(test,2,f))
test$X <- NULL
str(test)
str(test_missing)
test$Prediction <- predict(mylogit, newdata = test_missing, type="response")
submit <- data.frame(Id = test$Id, Predicted = test$Prediction)
write.csv(submit, file = "/Users/sayghosh/code/kaggle/test/criteo.csv", row.names = FALSE)

sum(test$Prediction>=.95)
test$Prediction[test$Prediction > 0.98]=.95
summary(test$Prediction)
#################################################
save.image("~/.RData")
save(mylogit, file = "/Users/sayghosh/code/kaggle/test/ctr.RData")
