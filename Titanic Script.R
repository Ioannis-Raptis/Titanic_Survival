# Load the dataset and the SDSFoundations package. 
# We added stringsAsFactors=FALSE so as not to create a NULL category in Embarked variable
dataset_train <- read.csv("data/train.csv", stringsAsFactors=FALSE)
library("SDSFoundations", lib.loc="C:/Program Files/R/R-3.4.3/library")

# View the column names of the dataset
names(dataset_train)

# View the Pclass distribution
table(dataset_train$Pclass)
round(prop.table(table(dataset_train$Pclass)), 3)
barplot(table(dataset_train$Pclass), main = "Passenger Class Distribution", xlab = "Class", ylab = "Count")

# View the Survived distribution
table(dataset_train$Survived)
tbl <- round(prop.table(table(dataset_train$Survived)), 3)
lbls <- paste(c("No", "Yes"), "\n", tbl * 100, "%", sep="")
pie(table(dataset_train$Survived), main = "Survival Distribution", labels = lbls)

# View the Sex distribution
table(dataset_train$Sex)
tbl <- round(prop.table(table(dataset_train$Sex)), 3)
lbls <- paste(c("Female", "Male"), "\n", tbl * 100, "%", sep="")
pie(table(dataset_train$Sex), main = "Sex Distribution", labels = lbls)

# View the Embarked distribution
table(dataset_train$Embarked)
tbl <- round(prop.table(table(dataset_train$Embarked)), 3)
lbls <- paste(c("No info", "Cherbourg", "Queenstown", "Southampton"), "\n", tbl * 100, "%", sep="")
pie(table(dataset_train$Embarked), main = "Embarked Distribution", labels = lbls)

# View the Age distribution
hist(dataset_train$Age, breaks = 20, main = "Histogram of Age", xlab = "Age")

# View the Fare distribution
hist(dataset_train$Fare, xlab = "Fare", ylab = "Count", main = "Histogram of Fare")

# Check for independence between variables sex and survived
tbl <- table(dataset_train$Sex, dataset_train$Survived)
chisq.test(tbl, correct = FALSE)$expected # Check that expected counts are over 5
chisq.test(tbl, correct = FALSE) # The variables are not independent...

# Convert variable Pclass from numerical to categorical
dataset_train$Pclass <- as.factor(dataset_train$Pclass)
levels(dataset_train$Pclass)

# Create a calculated field to use column Cabin in our model
dataset_train$HasCabin <- as.factor(ifelse (dataset_train$Cabin != "", "Yes", "No"))

# Check for independence between variables HasCabin and PClass
tbl <- table(dataset_train$HasCabin, dataset_train$Pclass)
chisq.test(tbl, correct = FALSE)$expected
chisq.test(tbl, correct = FALSE) # The variables are not independent.

# Subset the Dataset to exclude lines with no value for Age and Embarked
dataset_train_noNULLS <- dataset_train[complete.cases(dataset_train$Age) & dataset_train$Embarked != "",]
summary(dataset_train_noNULLS)

# Convert variable Embarked from string to categorical
dataset_train_noNULLS$Embarked <- as.factor(dataset_train_noNULLS$Embarked)
levels(dataset_train_noNULLS$Embarked)

# Create a logistic regression model to predict survival
mdl <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + HasCabin + Embarked, 
           data = dataset_train_noNULLS, family = "binomial")
library(car)
vif(mdl) # Test for multicolinearity. As expected PClass and HasCabin are somewhat correlated.
summary(mdl)

# We use the backwards elimination procedure to refine our model:
# we eliminate the variable with the highest p > 0.05 and check AIC in every iteration.

# Our final model. 
mdl <- glm(Survived ~ Pclass + Sex + Age + SibSp, data = dataset_train_noNULLS, family = "binomial")
vif(mdl)
summary(mdl)

# Plot of predicted probabilities of survival vs actual survival
plot(fitted.values(mdl), dataset_train_noNULLS$Survived, xlab = 'Predicted chance of survival', ylab = 'Actual Survival', 
     main = 'Plot of predicted probabilities of survival vs actual survival')

# Create survived hat column in our dataset
dataset_train_noNULLS$prob_of_survival <- fitted.values(mdl)
dataset_train_noNULLS$survived_hat <- ifelse(fitted.values(mdl) > 0.5, 1, 0)

# Create confusion matrix
confusion_matrix <- table(dataset_train_noNULLS$survived_hat, dataset_train_noNULLS$Survived)
round(prop.table(confusion_matrix), 2)

# Write the new dataset to an excel file, to create our CAP curve.
# library(openxlsx)
# write.xlsx(dataset_train_noNULLS,file = "dataset_train.xlsx")

#####################################
# Test our model against dataset_test and

# Import the test datasets.
dataset_test <- read.csv("data/test.csv")
test_survived_actuals <- read.csv("data/test survived actuals.csv")

# Convert Pclass to categorical in the test dataset
dataset_test$Pclass <- as.factor(dataset_test$Pclass)

# Fill out the missing values of age with the mean
m <- mean(dataset_test$Age,na.rm = T)
dataset_test$Age <- ifelse(is.na(dataset_test$Age), m,dataset_test$Age)

# Use our model to predict survival probabilities.
dataset_test$p_hat <- predict.glm(mdl,dataset_test,type = "response")

#  Compare predicted to actual survival
dataset_test$survived <- test_survived_actuals$Survived
dataset_test$survived_hat <- ifelse(dataset_test$p_hat > 0.5, 1, 0)

# Get our confusion matrix
confusion_matrix <- table(dataset_test$survived_hat, dataset_test$survived)
round(prop.table(confusion_matrix), 2)
