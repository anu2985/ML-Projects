setwd("/Users/anupama/Documents/Anupama Data Science/GreatLakes/Projects")
getwd()

# Split data into test and train datasets

attach(cars1)
table(cars1$Target)
sum(Target== 1)/nrow(cars1)

#13.73 is the percent of cars commuters. since its  biased data, we need to try different techniques:
# data partitioning and running the three models without any technique. checking the ratio of 1's targetted or in other words looking for sensitivity or True positive rate or recall where we are interested in determining the total 1's captured via the model from the actual ones.
# We, shall try SMOTE and running the logistic, KNN and Naive Bayes models
# Lastly will try bagging and boosting methods to see how the recall and model performance has improved.

#set.seed(300)
library(caTools)
#spl = sample.split(cars1, SplitRatio=0.70)
#trainLOG = subset(cars1, spl ==T)
#testLOG = subset(cars1, spl==F)


set.seed(400)
index<-createDataPartition(cars1$Target, p=0.7,list = FALSE,times = 1)
cars.train<-cars1[index,]
cars.test<-cars1[-index,]

#attach(cars.train)
prop.table(table(cars.train$Target))
prop.table(table(cars.test$Target))
#data is split in a ratio of 70:30 with train and test.

## Check split consistency
sum(cars.train$Target==1)/nrow(cars.train)
sum(cars.test$Target==1)/nrow(cars.test)
sum(Target==1) / nrow(cars1)

#the split should have similar percent of car usage Vs non car users and hence this achieved.

##LOGISTIC REGRESSION 

#  original model
LR_Train_model = glm(Target ~ ., data = cars.train, family= binomial)
summary(LR_Train_model)
car::vif(LR_Train_model)

#this is full model, which means including all the variables, the variables that show significant are Age, License Distance and the other variables like MBAs, Work.Exp are showing slight significant at a p-value of 0.05. But upon running the vif: Age, Work Exp, Salary, Distance and License - all turn out to be highly collinear. Hence, we need to drop some variables and re run the model. Full model : AIC 59.330 

#After dropping the non significant and highly correlated variables - Gender, Engineer , Work.Exp and Salary

LR_org.model1 = glm(Target ~ . -Age -Work.Exp, data = cars.train, family= binomial)
summary(LR_org.model1)
car::vif(LR_org.model1) 

LR_org.model2 = glm(Target ~ license + Distance + MBA  + Salary, data = cars.train, family= binomial)
summary(LR_org.model2)
car::vif(LR_org.model2)


LR_org.model3 = glm(Target ~ license + Distance  + Salary , data = cars.train, family= binomial)
summary(LR_org.model3)
car::vif(LR_org.model3)

#AIC has reduced to 56.94; also the variables have a VIF of 1 or closer to 1.

library(blorr) # to build and validate binary logistic models

#blr_step_aic_forward(LR_Train_model, details = FALSE)
#blr_step_aic_backward(LR_Train_model,details = TRUE)
#blr_step_aic_both(LR_Train_model, details = TRUE)


library(lmtest)
lrtest(LR_org.model3)

#with a significant P value, we can ensure that the logit model is valid.
#Also, lets check the Psuedo R2 or the goodness of fit; for which we consider the MC Fadden value.

# To get the logit R2 of goodness
#install.packages("pscl")
library(pscl)
pR2(LR_Train_model)
pR2(LR_org.model3)


#The MCFadden value of 0.66 with the original model for logistic interprets that the goodness of fit is a reasonably robust model.

# Trust only McFadden since its conservative
#if my McFadden > is between .0 to .10 - Goodness of fit is weak
#if my McFadden > is between .10 to .20 - Goodness of fit is fare
#if my McFadden > is between .20 to .30 - Goodness of fit is Moderately is robust
#if my McFadden > is between .30 and above - Goodness of fit is reasonably robust model
#Typical in non-linear model R2 will be less as against linear regression


org.odds = exp(coef(LR_org.model3))
write.csv(org.odds ,"org.odds_car.csv")


org.odds

#for identifying the relative importance of variables we have to use ODDS instead of PROB
prob=(org.odds[-1]/(1+org.odds[-1]))
prob

#with the above probability-with every unit increase in license by 1 and coefficient being 2.12; there is a 89% probability of an employee to use a car to commute to office; 
#With distance coefficient of 0.3049, with every unit increase in distance by 1, this is 57% probability that an employee would own a car to commute;
#Salary is also a highly significant variable and with every increase in Salary by one lakh, an a coeff of 0.162 the probability of owning a car to commute increases by 54.6% approximately;



relativeImportance=(org.odds[-1]/sum(org.odds[-1]))*100
relativeImportance[order(relativeImportance)]

#speaking about relative importance - the most important variables to consider are having a license,  Distance and Salary in chronological order.

#checking the confusion matrix for org model
predTest = predict(LR_org.model3, newdata= cars.test, type="response")
table(cars.test$Target, predTest>0.5)
# Accuracy of the model : 0.94 - 94.69%
# Recall/TPR/Sensitivity : 0.6667 - 66.67%
# Precision : 0.92 - 92.31%
# Specificity : 0.9912
# KS :85%
#Gini INdex : 0.823

################# Other methods for model performance ############

k = blr_gains_table(LR_org.model3,na.omit(cars.train))
plot(k)

blr_ks_chart(k, title = "KS Chart",
             yaxis_title = " ",xaxis_title = "Cumulative Population %",
             ks_line_color = "black")

blr_decile_lift_chart(k, xaxis_title = "Decile",
                      yaxis_title = "Decile Mean / Global Mean",
                      title = "Decile Lift Chart",
                      bar_color = "blue", text_size = 3.5,
                      text_vjust = -0.3)

blr_decile_capture_rate(k, xaxis_title = "Decile",
                        yaxis_title = "Capture Rate",
                        title = "Capture Rate by Decile",
                        bar_color = "blue", text_size = 3.5,
                        text_vjust =-0.3)

blr_confusion_matrix(LR_org.model3, data = cars.test)

blr_gini_index(LR_org.model3, data = na.omit(cars.test))

blr_roc_curve(k, title = "ROC Curve",
              xaxis_title = "1 - Specificity",
              yaxis_title = "Sensitivity",roc_curve_col = "blue",
              diag_line_col = "red", point_shape = 18,
              point_fill = "blue", point_color = "blue",
              plot_title_justify = 0.5)  

blr_rsq_mcfadden(LR_org.model3)
blr_rsq_mcfadden_adj(LR_org.model3)


#SMOTE MODEL- Logistic

cars.smote<-SMOTE(Target~., cars.train, perc.over = 250,perc.under = 150) #with this, there will be an equal split of 0.5 for both classes - 0 and 1
prop.table(table(cars.smote$Target))

#logistic model on balanced data
trainctrl<-trainControl(method = 'repeatedcv',number = 10,repeats = 3)
carsglm<- train(x= cars.smote[,c(1:8)],
                y=cars.smote[,9],
                method = "glm", 
                family = "binomial",
                trControl = trainctrl)

summary(carsglm$finalModel)
#running the model on the balance data; shows Age, Engineer1, MBA1, Distance are significant variables where as having a license is less significant;

carglmcoeff=exp(coef(carsglm$finalModel))
(carglmcoeff[-1]/(1+carglmcoeff[-1]))

# from the model; it definitely shows that Age, Males, License Holders and Engineers have a higher probability of impacting an employee's decision to use a car to commute office; where as Work Exp, Salary have a 44% and 49% to impact car usage for employees; whereas MBAs have only 4% chance to commute using car.

varImp(object = carsglm)
plot(varImp(object = carsglm), main="Variable Importance for Logistic Regression")
#from the variable importance - Age and Distance turn out to be the most significant variables; 

LR.smote.pred<-predict.train(object = carsglm,cars.test[,c(1:8)],type = "raw")
confusionMatrix(LR.smote.pred,cars.test[,9], positive='1')

smotepredTest = predict(carsglm, newdata= cars.test, type="prob")
smotepred= ifelse(smotepredTest[,2]>0.5, "1","0")
table(cars.test$Target, smotepredTest>0.5)
#confusionMatrix(as.factor(smotepred),cars.test[,9], positive='1')

# Accuracy of the model : 0.9545
# Recall/TPR/Sensitivity : 0.8333 83.33%
# Precision : 0.8333 83.33%
# Specificity: 0.9737

#Balancing the data set has slightly increased the accuracy from 94.69 to 95.45; it has bettered the recall or true positive rate as well from 69 to 83%, which means more more core users are captured from the model; where as my specificity which is my true negative rate is impacted- which has reduced from 99.12 to 97.37;


### KNN Model

library(ISLR)
library(caret)

set.seed(400)

#scaling for All Numeric variables
cars.KNN= scale(cars1[,c("Age", "Work.Exp","Salary","Distance")])

#Dummy code for Gender all factor levels
gender <- as.data.frame(dummy.code(cars1$Gender))
carsbinary<- as.data.frame(cars1[,c("Engineer","MBA","license","Target")])

cars.KNN= cbind(cars.KNN,gender,carsbinary)

#Training and Testing of the data
#Spliting data as training and test set. Using createDataPartition() function from caret

indx<- createDataPartition(y = cars.KNN$Target,p = 0.70,list = FALSE)

trainingKNN <- cars.KNN[indx,]
testingKNN<- cars.KNN[-indx,]

#Checking distibution in origanl data and partitioned data
prop.table(table(trainingKNN$Target)) * 100

prop.table(table(testingKNN$Target)) * 100

prop.table(table(cars.KNN$Target)) * 100

set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)

knnFit <- train(Target ~ ., data = trainingKNN, 
                method = "knn", 
                trControl = ctrl, 
                preProcess = c("center","scale"), 
                tuneLength = 20)

#Output of kNN fit
knnFit

plot(knnFit, main= "Accuracy with unbalanced sample")
#As per the model; accuracy of the model is highest when k= 41.

knnPredict <- data.frame( actual= testingKNN$Target,
                          predict(knnFit, newdata = testingKNN, type="prob"))
head(knnPredict)

knnPredict$pred= ifelse(knnPredict$X1> 0.5, 1, 0)

#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(table(knnPredict$pred, testingKNN$Target), positive = "1")

#Accuracy : 0.94 ,94.42%
#sensitivity: 0.61
#Precision (PPV): 1.00
#Specificity :1.00


#SMOTE Model -KNN1

ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)

knnFit.smote<- train(Target ~ ., data = smote.KNN, 
                method = "knn", 
                trControl = ctrl, 
                preProcess = c("center","scale"), 
                tuneLength = 20)

#Output of kNN fit
knnFit.smote

head(knnFit.smote$results, 5)

plot(knnFit.smote, main= "Accuracy with balanced sample")

#As per the smote model; accuracy of the model is highest when k= 11.

knnPredict <- data.frame( actual= testingKNN$Target,
                          predict(knnFit.smote, newdata = testingKNN, type="prob"))
head(knnPredict)

knnPredict$pred= ifelse(knnPredict$X1> 0.5, 1, 0)

#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(table(knnPredict$pred, testingKNN$Target), positive = "1")


#original sample k =41
#Accuracy : 0.94 ,94.42%
#sensitivity: 0.61
#Precision (PPV): 1.000
#specificity : 1.000

#smote sample k =11
#Accuracy : 0.90 - 90.9%
#sensitivity: 0.722 -72.22%
#Precision (PPV): 0.65 -65%
#specificity : 0.9386 -93.86%


#While using the unbalanced data, the accuracy reach when k= 41, where as in balanced data the highest accuracy reached when k=11; but comparing the accuracy is more while using the unbalanced data with 94.4%  which reduced at 90% when using the smote data; As the aim is always to maximize the 1's and not at the cost of reducing 0's, which in other words  in our problem statement would be truly capturing the actual car users and at the same time not falsely tagging the non car users are car users which would impact cost if in case any campaign has to be done to target car vs non car users. Sensitivity is higher with smoted data at 72% compared to 61% on unbalanced data. 










#KNN Model 2

#checking other method of KNN which used in telecom project


#removing Target variable from training and test datasets
trainKNN1 <- trainingKNN[,-10]
testKNN1 <- testingKNN[,-10]


#storing target variable for testing and training data as factor
cars_train_labels <- as.factor(trainingKNN$Target) 
dim(cars_train_labels)
cars_test_labels <- as.factor(testingKNN$Target)
dim(cars_train_labels)

library(class)
library(gmodels)

#KNN Model building
KNN_test_pred <- knn(train = trainKNN1, 
                     test = testKNN1, 
                     cl= cars_train_labels,
                     k = 3,
                     prob=TRUE)




CrossTable(x = cars_test_labels, y = KNN_test_pred,prop.chisq=FALSE, 
           prop.c = FALSE, prop.r = FALSE, prop.t = FALSE)
knn_tab <- table(KNN_test_pred,cars_test_labels)
knn_tab
1 - sum(diag(knn_tab)) / sum(knn_tab)   #0.45 Error rate


#install.packages("e1071")
library(e1071)

confusionMatrix(table(cars_test_labels,KNN_test_pred), positive = "1")

#KNN Model Performance

calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}


calc_class_err(actual    = cars_test_labels,
               predicted = KNN_test_pred) #0.045 Error rate

#set.seed(42)
k_to_try = 1:51
err_k = rep(x = 0, times = length(k_to_try))

for (i in seq_along(k_to_try)) {
  pred = knn(train = trainKNN1, 
             test = testKNN1, 
             cl= cars_train_labels, 
             k     = k_to_try[i])
  err_k[i] = calc_class_err(cars_test_labels, pred)
}

# plot error vs choice of k
plot(err_k, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "classification error",
     main = "(Test) Error Rate vs Neighbors")
# add line for min error seen
abline(h = min(err_k), col = "darkorange", lty = 3)
# add line for minority prevalence in test set
abline(h = mean(cars_test_labels == "1"), col = "grey", lty = 2)
#the orange dotted line is the minimum error line which represent the smallest observed test classification error rate. As the number of K increases the error rate increases to such the the error rate approaches the minority class prevalence in actual.
min(err_k)
#the minimum error rate is 0.030

#SMOTE Model - KNN

smote.KNN<-SMOTE(Target~., trainingKNN, perc.over = 250,perc.under = 150) #with this, there will be an equal split of 0.5 for both classes - 0 and 1
prop.table(table(smote.KNN$Target))


#removing Target variable from smote training data set for KNN
smote.train.KNN <- smote.KNN[,-10]



#storing target variable for testing and training data as factor
s_cars_train_labels <- as.factor(smote.KNN$Target) 
dim(s_cars_train_labels)

library(class)
library(gmodels)

#KNN Model building
smote_KNN_test_pred <- knn(train = smote.train.KNN, 
                     test = testKNN1, 
                     cl= s_cars_train_labels,
                     k = 3,
                     prob=TRUE)




CrossTable(x = cars_test_labels, y = smote_KNN_test_pred,prop.chisq=FALSE, 
           prop.c = FALSE, prop.r = FALSE, prop.t = FALSE)
knn_tab <- table(smote_KNN_test_pred,cars_test_labels)
knn_tab
1 - sum(diag(knn_tab)) / sum(knn_tab)   #0.45 Error rate


#install.packages("e1071")
#library(e1071)

confusionMatrix(table(cars_test_labels,smote_KNN_test_pred), positive = "1")

#KNN Model Performance

calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}


calc_class_err(actual    = cars_test_labels,
               predicted = smote_KNN_test_pred) #Error rate

set.seed(42)
k_to_try = 1:51
err_k = rep(x = 0, times = length(k_to_try))

for (i in seq_along(k_to_try)) {
  pred = knn(train = smote.train.KNN, 
             test = testKNN1, 
             cl= s_cars_train_labels, 
             k     = k_to_try[i])
  err_k[i] = calc_class_err(cars_test_labels, pred)
}

# plot error vs choice of k
plot(err_k, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "classification error",
     main = "(Test) Error Rate vs Neighbors")
# add line for min error seen
abline(h = min(err_k), col = "darkorange", lty = 3)
# add line for minority prevalence in test set
abline(h = mean(cars_test_labels == "1"), col = "grey", lty = 2)
#the orange dotted line is the minimum error line which represent the smallest observed test classification error rate. As the number of K increases the error rate increases to such the the error rate approaches the minority class prevalence in actual.
min(err_k)
#the minimum error rate is 0.030



#Naive Bayes

library(e1071)

cars.nb<-naiveBayes(x=cars.train[,1:8], y=cars.train[,9])
cars.nb$tables
cars.nb$apriori


pred_nb<-predict(cars.nb,newdata = cars.test[,1:8])

confusionMatrix(table(cars.test[,9],pred_nb), positive = "1")

#Accuracy :0.9545
#Sensitivity: 0.8750
#Precision :0.7778
#Specificity: 0.9655

#Naiv Bayes model explains the conditional probabilities of each other predictors in relation with the target; looking at the tables;
#for the numeric variables; explanation can be as follows :
# The mean Age for employees who use cars to commute was 35.6 years with a standard deviation of 3.21 years;
# The Average number of work experience years for car commuters were 15.7 years and non car commuters was 4.9 years, which could mean higher experienced people has more likelihood of using car to commute compared to young professionals
# Likewise, higher salary i.e. average salary of a car commuter was 36.5 lakhs per annum with a higher standard deviation of around 13.10 which also means; since this variable has outliers, the distribution is widely spread;
# Distance; Average distance for a person using a car to commute from home to office was 15.47 kms compared to average distance of non car users which was 10.83kms. Both Distant and closer living population didn't have  wider gap in terms of their standard deviation which ranged between 3.2 and 3.67 for car and non car commuters resp.

#for the binary variables: explanation can be as follows:
# Looking at the gender of the commuters - 83% males used car to commute where only 16% females used car to commute to office;
#secondly, out of the 13% car commuters- nearly 86% were engineers and only 18% were MBA's;
#lastly but not least, 83% license holders used car to commute;

#SMOTE Model- Naive Bayes


cars.nb.smote<-naiveBayes(x=cars.smote[,1:8], y=cars.smote[,9])
cars.nb.smote$tables
cars.nb.smote$apriori


pred_nb_smote<-predict(cars.nb.smote,newdata = cars.test[,1:8])

confusionMatrix(table(cars.test[,9],pred_nb_smote), positive = "1")

#Accuracy: 0.9545
#sensitivity : 0.8750
#Precision: 0.7778
#Specificity: 0.9655

#the smote model has not improved any of the metrics in as compared to model created with the unbalanced and original dataset. Moreover, the conditional probabilities created would also not be considered useful because the  probabilities are not actual in terms of the target and predictors variable. Hence the naive bayes is not a very useful model here.



