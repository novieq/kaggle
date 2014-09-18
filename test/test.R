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
classes <- sapply(train, class)
classes
train = read.csv("/Users/sayghosh/code/kaggle/test/train_random.csv", colClasses = classes)
print(Sys.time())

str(train)
summary(train)
train$Id<-NULL
train$Label<-as.factor(train$Label)

# Build a deeper tree
print(Sys.time())
fit <- rpart(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13, data=train, method="class")
print(Sys.time())

# Plot it with base-R
plot(fit)
text(fit)
# And then make it look better with fancyRpartPlot!
test <- read.csv("/Users/sayghosh/code/kaggle/test/test.csv")
print(Sys.time())

# Now let's make a prediction and write a submission file
classes <- sapply(test, class)
classes
Prediction <- predict(fit, test, type = "prob")

submit <- data.frame(Id = test$Id, Label = Prediction[,1])
write.csv(submit, file = "/Users/sayghosh/code/kaggle/test/criteo.csv", row.names = FALSE)
print(Sys.time())


mylogit <- glm(Label ~ I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10 + I11 + I12 + I13 , data = train, family = "binomial")
summary(mylogit)
Prediction <- predict(mylogit, newdata = test)
summary(Prediction)
help(predict)
