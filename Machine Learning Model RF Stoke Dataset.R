#Building a step wise step Machine Learning Model

#Exploratory Data Analysis (EDA)

#Here we are using the Stroke Prediction Dataset from Kaggle to make predictions. Our task is to examine existing patient records in the training set and use that knowledge to predict whether a patient in the evaluation set is likely to have a stroke or not.

#---------------#------------------------------------#---------------------

library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
#Read and Import the data

df_stroke <- read.csv("~/Downloads/healthcare-dataset-stroke-data.csv")


#Undertanding the data

glimpse(df_stroke)

summary (df_stroke)


count(df_stroke, gender == 'Other')  

df_stroke<- df_stroke[df_stroke$gender != 'Other',]


#Now we have new dataframe containing only the male and female dataset


#Now we will check the NA in the table and drop those rows so that there is no error encountered later creating the algorithm

sum(df_stroke$bmi == 'N/A')

#We can see threre are 201 enteries in bmi column which is empty, this 201 entries consisit of 5% of the total data so I would replace it with mean of the bmi column instead of just dropping the column

#Imputing datatset
df_stroke$bmi <- as.numeric(df_stroke$bmi)# converting the datatype to numeric as it was in character before

df_stroke$bmi[is.na(df_stroke$bmi)] <- mean(df_stroke$bmi, na.rm = TRUE)

glimpse(df_stroke)

sum(is.na(df_stroke))#seems like now there is no missing value and we can go ahead 

#Now we have to convert each charatcer column to factor to implement machine learning algorith in future
library(dplyr)

df_stroke <- df_stroke %>%
  mutate_if(is.character, as.factor)

view(df_stroke)




#Now we will generate some graphs

p1<- ggplot(df_stroke, aes(x= "", y = gender, fill = gender)) +geom_bar(stat = "identity", width = 1)+
  coord_polar("y", start = 0)+theme_minimal()


# calculate frequency of heart disease values
heart_disease <- df_stroke %>%
  count(heart_disease)

# create pie chart
p2<-ggplot(heart_disease, aes(x = "", y = n, fill = factor(heart_disease)))+
  geom_bar(stat = "identity", width = 1)+
  coord_polar("y", start = 0)+
  theme_minimal()+
  labs(fill = "Heart Disease")+
  scale_fill_manual(values = c("#FF9999", "#66CCFF"))+
  ggtitle("Heart Disease Pie Chart")


# calculate frequency of hypertension values
hypertension <- df_stroke %>%
  count(hypertension)

p3<- ggplot(hypertension, aes(x = "", y = n, fill = factor(hypertension)))+
  geom_bar(stat = "identity", width = 1)+
  coord_polar("y", start = 0)+
  theme_minimal()+
  labs(fill = "Hypertension")+
  scale_fill_manual(values = c("#FF9999", "#66CCFF"))+
  ggtitle("Hypertension Pie Chart")

p4<- ggplot(df_stroke, aes(x = "", y = ever_married, fill = ever_married)) + 
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  theme_minimal() 

p5<- ggplot(df_stroke, aes(x = "", fill = work_type)) +
  geom_bar(width = 1, color = "white") +
  coord_polar(theta = "y") +
  theme_void() +
  labs(fill = "Work Type")




#######----------------#-#-----------------------------#####


#Model Building and Prediction

#We can create a few additional bar charts to see how each of these variables relates to the target variable, which is the stroke possibility for the individual.


#Let's split the final data set into training and test data set

n_obs<- nrow(df_stroke)
split<- round(n_obs * 0.7)

train<- df_stroke [1:split,] 

#Create Test

test<- df_stroke[(split +1): nrow(df_stroke),] 

dim(train)
dim(test)

train$stroke <- as.factor(train$stroke)
test$stroke <- as.factor(test$stroke)


#We use Random Forest algorithm for this problem as it is normally used in supervised learning since our problem has only two possible outcomes.

#Modeling

rf_model<-randomForest(formula= stroke~.,data = train) 
rf_model

#Out-of-Bag (OOB) estimate of error rate (7.13%), the number of trees (500), the variables at each split (3), and the function used to build the classifier (randomForest). We must evaluate the model’s performance on similar data once trained on the training
##set. We will make use of the test dataset for this. Let us print the confusion matrix to see how our classification model performed on the test data –
# Check levels of stroke in train and test datasets
levels(train$stroke)
levels(test$stroke)

# Ensure that both datasets have the same levels for stroke factor variable
test$stroke <- factor(test$stroke, levels = levels(train$stroke))


confusionMatrix(predict(rf_model, test), test$stroke)


#We can see that the accuracy is nearly 100% with a validation dataset, suggesting that the model was trained well on the training data.

#The confusion matrix shows the performance of the random forest model on the test dataset. The rows correspond to the predicted classes (0 and 1) and the columns correspond to the actual classes.

##The confusion matrix shows that out of 1533 instances of class 0, the model correctly predicted all of them as class 0. However, out of 2 instances of class 1, the model incorrectly predicted them as class 0.

###The accuracy of the model is calculated as (number of correct predictions)/(total number of predictions), which in this case is (1531+0)/(1531+0+2+0) = 0.9987, or 99.87%. This means that the model is very accurate at predicting the absence of stroke (class 0), but not very good at predicting the presence of stroke (class 1).

###The other statistics in the confusion matrix such as Sensitivity, Specificity, Pos Pred Value, and Neg Pred Value are not calculated because there are no true positives, true negatives, false positives, or false negatives for class 1.
