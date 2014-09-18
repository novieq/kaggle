
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

###################################################
## Step 1: Data Preparation
###################################################

train <- read.csv("/Users/sayghosh/code/kaggle/Titanic/train.csv")

# Examine structure of dataframe
str(train)

# Look at number of people who survived
table(train$Survived);
prop.table(table(train$Survived));

train$Pclass = as.factor(train$Pclass);
train$Survived = as.factor(train$Survived);
train$PassengerId = NULL;
train$Name=NULL;

#####################################################
## Stage 2: Model building, using Decision Trees 
#####################################################

# Build a deeper tree
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, 
             method="class",
             control=rpart.control(minsplit=5, cp=0.005)))
#By adding the control and the pruning parameter in the decision tree there is no improvement in the score

# Plot it with base-R
plot(fit)
text(fit)
# And then make it look better with fancyRpartPlot!
fancyRpartPlot(fit)
fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
new.fit <- prp(fit,snip=TRUE)$obj
fancyRpartPlot(new.fit)
#####################################################
## Stage 3: Predcit label for test data
#####################################################

#Load test data
test <- read.csv("/Users/sayghosh/code/kaggle/Titanic/test.csv")

#Change data types
test$Pclass = as.factor(test$Pclass);
# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "titanic.csv", row.names = FALSE)

