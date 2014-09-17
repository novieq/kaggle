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
initial <- read.csv("/Users/sayghosh/code/kaggle/test/train_random.100000.csv", nrows = 100)
classes <- sapply(initial, class)
classes
train = read.csv("/Users/sayghosh/code/kaggle/test/train_random.100000.csv", colClasses = classes, nrows=100001, comment.char="")
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
train_[,16:41] <- train[16:41]
train_[,1:2] <- train[,1:2]
str(train)
str(train_)
summary(train)
summary(train_)

mylogit <- glm(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13 , data = train_, family = "binomial", na.action = "na.omit")
summary(mylogit)


# 3. Name : Using only factor variable
#    Error :  0.5404027 , 0.5403759, 0.5403125, 0.5385134
#    Variable : C1, C2, C3, C4, C5, C6
# By increasing size of the training sample, the CV error does not reduce, let us work on adding more features
train_=data.frame(apply(train,2,f))
train_[,16:41] <- train[16:41]
train_[,1:2] <- train[,1:2]
str(train)
str(train_)
summary(train)
summary(train_)

mylogit <- glm(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13 + C1 + C2 + C3 + C4 + C5, data = train_, family = "binomial", na.action = "na.omit")
summary(mylogit)

###################################################
## Step 3: Model Cross Validation 
###################################################

print(Sys.time())
test_missing_cv <- read.csv("/Users/sayghosh/code/kaggle/test/test_cv.csv", colClasses = classes, nrows=1000001)
print(Sys.time())

test_cv=data.frame(apply(test_missing_cv,2,f))
test_cv$Prediction <- predict(mylogit, newdata = test_cv, type="response")
cv_error <- logarithmicLogFun( test_cv$Label, test_cv$Prediction)
cv_error

###################################################
## Step 4: Predicting on test set
###################################################

initial <- read.csv("/Users/sayghosh/code/kaggle/test/test.csv", nrows = 100)
classes <- sapply(initial, class)
classes
print(Sys.time())
test <- read.csv("/Users/sayghosh/code/kaggle/test/test.csv", colClasses = classes, nrows=6042136)
print(Sys.time())

#Remove missing values from test data
test_missing=data.frame(apply(test,2,f))

test$Prediction <- predict(mylogit, newdata = test_missing, type="response")
submit <- data.frame(Id = test$Id, Predicted = test$Prediction)
write.csv(submit, file = "/Users/sayghosh/code/kaggle/test/criteo.csv", row.names = FALSE)

