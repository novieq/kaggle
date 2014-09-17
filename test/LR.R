###################################################
## Step 1: Data Preparation
###################################################
rm(list = ls(all = TRUE))
gc()

library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
# Load Training data
print(Sys.time())
initial <- read.csv("/Users/sayghosh/code/kaggle/test/train_random.csv", nrows = 100)
classes <- sapply(initial, class)
classes
train = read.csv("/Users/sayghosh/code/kaggle/test/train_random.csv", colClasses = classes)
print(Sys.time())

str(train)
summary(train)
train$Id<-NULL
train$Label<-as.factor(train$Label)

# And then make it look better with fancyRpartPlot!
initial <- read.csv("/Users/sayghosh/code/kaggle/test/test.csv", nrows = 100)
classes <- sapply(initial, class)
classes

test <- read.csv("/Users/sayghosh/code/kaggle/test/test.csv", colClasses = classes)
print(Sys.time())

nrow(train)
summary(train)

mylogit <- glm(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13 , data = train, family = "binomial", na.action = "na.omit")
summary(mylogit)
test$Prediction <- predict(mylogit, newdata = test, type="response")
submit <- data.frame(Id = test$Id, Label = test$Prediction)
write.csv(submit, file = "/Users/sayghosh/code/kaggle/test/criteo.csv", row.names = FALSE)
