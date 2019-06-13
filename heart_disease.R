# Data Science Capstone Choose Your Own

rm(list=ls())

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org"); library(tidyverse)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org"); library(caret)
if(!require("ROCR")) install.packages("ROCR", repos = "http://cran.us.r-project.org"); library(ROCR)

# Load data 
data <- read.csv('/Users/luke/Documents/Rstudio/heart_disease_ml/heart.csv')
names(data)[1] <- 'age'

# Converting to factors
data$sex <- as.factor(data$sex)
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)
data$ca <- as.factor(data$ca)
data$thal <- as.factor(data$thal)
data$target <- as.factor(data$target)


#### Exploratory Analyses ####

# Check for blank values in the dataset
sapply(data, function(x) sum(is.na(x))) # No NA values

# View summary data 
summary(data)

# Bivariate between independent variables and target
barplot(table(data$target, data$sex), 
        main = 'Split of Target by Sex Buckets', 
        xlab = 'Sex', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=1,y=200),  col=c("red", "blue"))

barplot(table(data$target, data$cp), 
        main = 'Split of Target by cp buckets', 
        xlab = 'cp', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=5,y=150),  col=c("red", "blue"))

barplot(table(data$target, data$fbs), 
        main = 'Split of Target by fbs buckets', 
        xlab = 'fbs', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=2.2,y=200),  col=c("red", "blue"))

barplot(table(data$target, data$restecg), 
        main = 'Split of Target by restecg buckets', 
        xlab = 'restecg', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=4,y=150),  col=c("red", "blue"))

barplot(table(data$target, data$exang), 
        main = 'Split of Target by exang buckets', 
        xlab = 'exang', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=2.2,y=200),  col=c("red", "blue"))

barplot(table(data$target, data$slope), 
        main = 'Split of Target by slope buckets', 
        xlab = 'slope', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=1.25,y=120),  col=c("red", "blue"))

barplot(table(data$target, data$ca), 
        main = 'Split of Target by ca buckets', 
        xlab = 'ca', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'),  col=c("red", "blue"))

barplot(table(data$target, data$thal), 
        main = 'Split of Target by thal buckets', 
        xlab = 'thal', ylab = 'Count', 
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=2,y=150),  col=c("red", "blue"))

p1 <- hist(data$age[data$target==0], col=c("red")) 
p2 <- hist(data$age[data$target==1], col=c("blue"))
plot(p1, col="red", xlim=c(0,100), main='Frequency Distribution of Age by Target Buckets', xlab='Age')
plot(p2, col="blue", xlim=c(0,100), add=T)

p1 <- hist(data$trestbps[data$target==0], col=c("red")) 
p2 <- hist(data$trestbps[data$target==1], col=c("blue"))
plot(p2, col="red", xlim=c(80,200), main='Frequency Distribution of Trestbps by Target Buckets', xlab='Trestbps')
plot(p1, col="blue", xlim=c(80,200), add=T)

p1 <- hist(data$chol[data$target==0], col=c("red")) 
p2 <- hist(data$chol[data$target==1], col=c("blue"))
plot(p2, col="red", xlim=c(100,600), main='Frequency Distribution of Chol by Target Buckets', xlab='Chol')
plot(p1, col="blue", xlim=c(100,600), add=T)

p1 <- hist(data$thalach[data$target==0], col=c("red")) 
p2 <- hist(data$thalach[data$target==1], col=c("blue"))
plot(p2, col="red", xlim=c(50,250), main='Frequency Distribution of Thalach by Target Buckets', xlab='Thalach')
plot(p1, col="blue", xlim=c(50,250), add=T)

p1 <- hist(data$oldpeak[data$target==0], col=c("red")) 
p2 <- hist(data$oldpeak[data$target==1], col=c("blue"))
plot(p2, col="red", xlim=c(0,8), main='Frequency Distribution of Oldpeak by Target Buckets', xlab='Oldpeak')
plot(p1, col="blue", xlim=c(0,8), add=T)


#### Create train and test sets ####
set.seed(1)
test_index <- createDataPartition(y = data$target, times = 1, p = 0.2, list = FALSE)
train_data <- data[-test_index,]
test_data <- data[test_index,]


#### Building regression model #### 
model <- glm(target~., data = train_data, family = binomial(link = 'logit'))
summary(model)

# Stepwise backward elimination to select variables and improve model
select_vars_model <- step(model)
summary(select_vars_model)


#### Making predictions on test set and evaluating performance ####
test_predicted <- predict(select_vars_model, test_data, type='response')

ROCRpred = prediction(test_predicted,test_data$target)
ROCRperf = performance(ROCRpred, "tpr", "fpr")

# Plot ROC curve, calculate AUC
plot(ROCRperf, main='ROC curve')
auc <- attributes(performance(ROCRpred, 'auc'))$y.values[[1]]

# Confusion Matrix
predClass <- as.factor(ifelse(test_predicted>=0.5,1,0))
confusionMatrix(test_data$target, predClass)