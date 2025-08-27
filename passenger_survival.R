#GOAL:  Make a logistic regression model using
# Titanic data to predict which passengers are likely to survive

#Data includes:
#PassengerId : Row ID in the data set
#Survived : If a passenger survived or not (0 = No, 1 = Yes)
#Pclass : Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
#Name : Passenger’s name
#Sex : Passenger’s gender (male or female)
#Age : Passenger’s age (in years)
#SibSp : # of siblings / spouses aboard the Titanic
#Parch : # of parents / children aboard the Titanic
#Ticket : Ticket number
#Fare : Passenger fare
#Cabin : Cabin number
#Embarked : Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)



#load libraries and data
install.packages(c("tidyverse", "titanic"))
library(tidyverse)
library(titanic)

titanic <- titanic::titanic_train

# Keep useful columns only

# 1. From exploratory analysis, found that the range of values
#for Ticket and Cabin are very broad and will not be usable
#predictors.
# 2. PassengerId and Name don't give useful information for predicting survival

titanic <- titanic %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

#Age has 177 missing values, replace them by median age based on sex

titanic <- titanic %>%
  group_by(Pclass) %>%
  mutate(Age = if_else(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()

#Embarked has 2 empty values, remove them
titanic <- titanic %>%
  filter(Embarked != "")

#Convert catagorical columns to factors
titanic <- titanic %>% mutate(
    Survived = factor(Survived, labels = c("No", "Yes")),
    Pclass = factor(Pclass),
    Sex = factor(Sex),
    Embarked = factor(Embarked)
  )

#Create new feature called family size

titanic <- titanic %>%
  mutate(FamilySize = SibSp + Parch)

#Data is split between ~40% survived ~60% did not survived
#This is balanced enough to proceed but can balance by under
#or oversampling if needed

#split data for training and testing with 80% training

set.seed(42)
train_index <- sample(seq_len(nrow(titanic)), size = 0.8 * nrow(titanic))
train <- titanic[train_index, ]
test <- titanic[-train_index, ]

#build the logistic regression model
model <- glm(Survived ~ Pclass + Sex + Age + Fare + FamilySize + Embarked,
             data = train, family = binomial)


#Because the prediction results in the logistic regression model
#are in the form of probabilities, we must convert these values
#into our target category/class using threshold value. Any values
#above the threshold value will be classified as positive class.
#By default, the threshold value is 0.5.
pred_probs <- predict(model, test, type = "response")
pred <- if_else(pred_probs > 0.5, "Yes", "No") %>% factor(levels = c("No", "Yes"))


#calculate accuracy
conf_matrix <- table(Predicted = pred, Actual = test$Survived)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", round(accuracy * 100, 2), "%\n")
