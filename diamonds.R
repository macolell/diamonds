my_packages <- c('readxl', 'readr', 'tidyr', 'dplyr', 'lfe', 'janitor', 'tictoc', 'pscl', 'plyr')
lapply(my_packages, require, character.only = T)
ml_packages <-c('class', 'gmodels', 'pROC', 'rpart', 'ranger', 'rpart.plot', 'caTools', 'ROCR', 'gbm', 'caret', 'xgboost', 'glmnet')
lapply(ml_packages, require, character.only = T)
set.seed(1234)
#EDA to understand the data 
diamonds <- read_csv('/Users/michaelcolellajensen/Desktop/Data Sets/diamonds.csv')
diamonds <- diamonds%>%
  mutate(cut = as.factor(cut),
         color = as.factor(color),
         clarity = as.factor(clarity))
str(diamonds)
summary(diamonds)
sum(is.na(diamonds))
mean(diamonds$price)

hist(diamonds$price)
hist(diamonds$carat)

ggplot(diamonds, aes(x=color))+
  geom_bar(fill = c('#800000', '#B22222', '#DC143C', '#FF6347', '#CD5C5C', '#FA8072', '#FF4500'))
ggplot(diamonds, aes(x=cut))+
  geom_bar(fill = c('#800000', '#DC143C', '#FF6347', '#CD5C5C', '#FA8072'))
ggplot(diamonds, aes(x=clarity))+
  geom_bar(fill = c('#800000', '#B22222', '#DC143C', '#FF6347', '#CD5C5C', '#FA8072', '#FF4500', '#F08080'))

#test train split 
split= sample.split(diamonds$price, SplitRatio = 0.8)
train<- subset(diamonds, split == TRUE)
test <- subset(diamonds, split == FALSE)

#Cross validation parameter tuning for a GBM model 
#Using random search for hyperparams
cv_control<- trainControl(method='cv', number=3, search = 'random')
tic()
gbm_cv_model <- train(price~carat+cut+color+clarity+depth+table,
                   data=train, 
                   tuneLength = 10, 
                   method = 'gbm',
                   trControl = cv_control)
toc()
print(gbm_cv_model)

#We will use the results of the random search for our model 
#ntrees=1697, depth=10, shrinkage = 0.084, min node size =12
tic()
gbm <- gbm(price~carat+cut+color+clarity+depth+table, 
           data = train,
           n.trees = 1697,
           interaction.depth = 10, 
           shrinkage = 0.084, 
           n.minobsinnode = 12)
toc()
summary(gbm)
#Generate predictions and evaluate RMSE and r2
gbm_predictions = predict(gbm, newdata = test)
residual <- test$price - gbm_predictions
oos_rmse <- sqrt(mean(residual^2))
print(oos_rmse)

SSE = sum((test$price-gbm_predictions)^2)
SST = sum((test$price- mean(train$price))^2)
oos_r2<-1 - SSE/SST
print(oos_r2)

###Attempting to use XGBoost for another model 
require(vtreat)
require(magrittr)
treatplan<- designTreatmentsZ(diamonds, vars, verbose=FALSE)
vars <- c('carat', 'cut', 'color', 'clarity', 'depth', 'table')
(newvars <- treatplan %>%
    use_series(scoreFrame)%>%
    filter(code %in% c('clean', 'lev'))%>%
    use_series(varName))

diamonds_treat <- prepare(treatplan, diamonds, varRestriction = newvars)
diamonds_treat <- diamonds_treat%>%
  mutate(carat = diamonds$carat,
         depth = diamonds$depth,
         table = diamonds$table)

xgb_split= sample.split(diamonds_treat$cut_lev_x_Ideal, SplitRatio = 0.8)
xgb_train<- subset(diamonds_treat, split == TRUE)
xgb_test <- subset(diamonds_treat, split == FALSE)

xgb_control <- trainControl(method='cv', number=3, search = 'random')
tic()
xgb_cv_model<- train(y~.,
                     data = as.matrix(xgb_train),
                     tuneLength = 10,
                     method = 'xgbTree',
                     trControl = xgb_control)
toc()
print(xgb_cv_model)
#use the random search results to build the xgboost model 
tic()
xgb<-xgboost(data=as.matrix(xgb_train),
             label = train$price,
             nrounds = 937,
             max_depth =7,
             eta=0.075,
             gamma = 1.46, 
             colsample_bytree = 0.65,
             min_child_weight = 15, 
             subsample = 0.68)
toc()
summary(xgb)
xgb_predictions <- predict(xgb, newdata= as.matrix(xgb_test))
xgb_residual <- test$price - xgb_predictions
xgb_rmse <- sqrt(mean(xgb_residual^2))
print(xgb_rmse)

xgb_SSE = sum((test$price-xgb_predictions)^2)
xgb_SST = sum((test$price- mean(train$price))^2)
xgb_r2<-1 - xgb_SSE/xgb_SST
print(xgb_r2)

#using a GLMNET model 
glm_cv_control <- trainControl(method = 'cv', 
                               number = 3, 
                               verboseIter = T)
tic()
glmnet_model<- train(price~carat+cut+color+clarity+depth+table,
                     data= train,
                     metric = 'RMSE', 
                     method='glmnet',
                     tuneLength = 10,
                     trControl = glm_cv_control)
toc()
print(glmnet_model)
#model found best parameters as alpha = 0.1, more lasso, and lambda = 3.55 low penalty
#train RMSE is 826 and r2 is 91.6

glmnet_predictions <- predict(glmnet_model, newdata = test)
glmnet_residual <- test$price - glmnet_predictions
glmnet_rmse <- sqrt(mean(glmnet_residual^2))
print(glmnet_rmse)

glm_SSE = sum((test$price-glmnet_predictions)^2)
glm_SST = sum((test$price- mean(train$price))^2)
glmnet_r2<-1 - glm_SSE/glm_SST
print(glmnet_r2)

reg <- lm(price~carat+cut+color+clarity+depth+table,
          data= train)
summary(reg)
reg_predictions <- predict(reg, newdata = test)
reg_residual <- test$price - reg_predictions
reg_rmse <- sqrt(mean(reg_residual^2))
print(reg_rmse)

reg_SSE = sum((test$price-reg_predictions)^2)
reg_SST = sum((test$price- mean(train$price))^2)
reg_r2<-1 - reg_SSE/reg_SST
print(reg_r2)

#clear enviorment
rm(list = ls())
cat("\014")
lapply(my_packages, detach, character.only = T)
lapply(ml_packages, detach, character.only = T)

