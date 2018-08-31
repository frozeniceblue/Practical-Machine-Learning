library(caret)
library(rpart)
library(randomForest)
library(kernlab)
library(caTools)
set.seed(777)
## load the data
train=read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),
               na.strings=c("NA","#DIV/0!",""))
test=read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),
              na.strings=c("NA","#DIV/0!",""))
head(train)
head(test)
dim(train)
dim(test)

## cleaning the data
delete_var=nearZeroVar(train,saveMetrics=TRUE)
train=train[,!delete_var$nzv]
train=train[,-(1:6)]
na_var=sapply(colnames(train), function(x)
              if(sum(is.na(train[,x]))>0.5*nrow(train))
                {return(TRUE)
                }else{
                return(FALSE)
                }
              )
train=train[,!na_var]

## cross validation & fitted model
split=createDataPartition(y=train$classe, p=0.7, list=FALSE)
train_train=train[split,]
train_test=train[-split,]

trControl=trainControl(method = "none", verboseIter=FALSE, allowParallel=TRUE)
SVMRadial=train(classe ~ ., data = train_train, method = "svmRadial", trControl= trControl)
RandomForest=train(classe ~ ., data = train_train, method = "rf", trControl= trControl)
LogitBoost=train(classe ~ ., data = train_train, method = "LogitBoost", trControl= trControl)
DecisionTree=rpart(classe ~ ., data=train_train, method="class")

## results and Test
predSVM=predict(SVMRadial,train_test)
confusionMatrix(predSVM, train_test$classe)

predRF=predict(RandomForest,train_test)
confusionMatrix(predRF, train_test$classe)

predLB=predict(LogitBoost,train_test)
confusionMatrix(predLB, train_test$classe)

predDT=predict(DecisionTree, train_test, type = "class")
confusionMatrix(predDT, train_test$classe)

modelname=c("Random Forest", "SVM (radial)","LogitBoost","Decision Tree")
accuracy=c(confusionMatrix(predSVM, train_test$classe)$overall["Accuracy"],
           confusionMatrix(predRF, train_test$classe)$overall["Accuracy"],
           confusionMatrix(predLB, train_test$classe)$overall["Accuracy"],
           confusionMatrix(predDT, train_test$classe)$overall["Accuracy"])
Result=cbind(modelname, accuracy)

knitr::kable(Result)

## Prediction
final_var=colnames(train_train[,-length(train_train)])
test_test=test[final_var]

predTest=predict(RandomForest,test_test)
predTest

## submit the file
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predTest)




