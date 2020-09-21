
getwd()
#install.packages("dplyr")
#install.packages("gmodels")
#install.packages("naivebayes")

# Libraries for ML models
library(class)
#libraries for plotting
library(gmodels)
#library(datasets)



#Reading the cell phone data for KNN
celldataKNN= celldata
str(celldataKNN)
celldataKNN= as.data.frame(celldataKNN)
dim(celldataKNN)
attach(celldataKNN)
#Scaling is an important aspect for running a KNN model else the model would bias itself to higher numeric values.

#scaling for All Numeric variables
celldataKNN[,c("AccountWeeks", "DataUsage","DayMins","DayCalls","MonthlyCharge","OverageFee","RoamMins")]= scale(celldataKNN[,c("AccountWeeks", "DataUsage","DayMins","DayCalls","MonthlyCharge","OverageFee","RoamMins")])

#Dummy code for Customer Service Calls as there are 10 levels
CustServCalls <- as.data.frame(dummy.code(celldataKNN$CustServCalls))

celldataKNN= cbind(celldataKNN,CustServCalls)
celldataKNN= celldataKNN[,-6]



#Random splitting of iris data as 70% train and 30%test datasets
set.seed(123)
ind <- sample(2, nrow(celldataKNN), replace=TRUE, prob=c(0.7, 0.3))
trainKNN <- celldataKNN[ind==1,]
testKNN <- celldataKNN[ind==2,]
#checking the dimensions of train and test datasets
dim(trainKNN)
dim(testKNN)

#removing Target variable from training and test datasets
trainKNN1 <- trainKNN[,-1]
testKNN1 <- testKNN[,-1]


#storing target variable for testing and training data as factor
cell_train_labels <- as.factor(trainKNN$Churn) 
dim(cell_train_labels)
cell_test_labels <- as.factor(testKNN$Churn)
dim(cell_train_labels)

#KNN Model building
KNN_test_pred <- knn(train = trainKNN1, 
                      test = testKNN1, 
                      cl= cell_train_labels,
                      k = 3,
                      prob=TRUE)


# library(gmodels)

CrossTable(x = cell_test_labels, y = KNN_test_pred,prop.chisq=FALSE, 
           prop.c = FALSE, prop.r = FALSE, prop.t = FALSE)
knn_tab <- table(KNN_test_pred,cell_test_labels)
knn_tab
1 - sum(diag(knn_tab)) / sum(knn_tab)   ## Error
# Error when k=3 = 10.8% : 89.1%  #sensitivity 0.43 #specificity 0.968
# Error when k=5 = 9.3% : 90.6%  #sensitivity 0.42 specificity 0.988
# Error when k=7 = 9.4% : 90.5%  #sensitivity 0.39 specificity 0.992
# Error when k=9 = 10.1%% : 89.8% #sensitivity 0.35 specificity 0.989
# Error when k=21 = 10.7%% :89.2% #sensitivity 0.28 specificity 0.990
install.packages("e1071")
library(e1071)

confusionMatrix(table(cell_test_labels,KNN_test_pred), positive = "1")

#KNN Model Performance

calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}


calc_class_err(actual    = cell_test_labels,
               predicted = KNN_test_pred) #10.98 Error rate

set.seed(42)
k_to_try = 1:21
err_k = rep(x = 0, times = length(k_to_try))

for (i in seq_along(k_to_try)) {
  pred = knn(train = trainKNN1, 
             test = testKNN1, 
             cl= cell_train_labels, 
             k     = k_to_try[i])
  err_k[i] = calc_class_err(cell_test_labels, pred)
}

# plot error vs choice of k
plot(err_k, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "classification error",
     main = "(Test) Error Rate vs Neighbors")
# add line for min error seen
abline(h = min(err_k), col = "darkorange", lty = 3)
# add line for minority prevalence in test set
abline(h = mean(cell_test_labels == "1"), col = "grey", lty = 2)
#the orange dotted line is the minimum error line which represent the smallest observed test classification error rate. As the number of K increases the error rate increases to such the the error rate approaches the minority class prevalence in actual.
min(err_k)

#Accuracy reduces as the number of 'k' increases. But at the same time its, important to see the True Positive Rate and False negative rate of the model. 
#Known that 14.4% is my churn rate or in other words, customers who have canceled the service;  
# As per the problem statement, we need to predict the customers who are at the verse at canceling the service so that we could offer them plans and retain them. 
# If K=3 is selected, the accuracy is 89% and error rate is 10%, in which the True Positive rate is 0.43 which highest compared to any of the other models with higher number of K. As the aim of the model is to predict the customers who will cancel the service, we should aim at increasing the TPR or sensitivity. which helps to more accurately capture customers who have in actual churned. 



