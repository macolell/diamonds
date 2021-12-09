#clear environment
rm(list=ls())
cat("\014")

#load packages
library(caret)

set.seed(1234)

#load data
diamond=read.csv("mod_diamonds.csv")
str(diamond)

#choose a random sample
diamondsub<-diamond[sample(nrow(diamond),1000,replace=TRUE),]

#Normalize data
preproc = preProcess(diamondsub)
diamondsubNorm = predict(preproc, diamondsub)
mean(diamondsubNorm$table)
sd(diamondsubNorm$table)

## HIERARCHICAL CLUSTERING
Diamond.Dist = dist(diamondsubNorm, method = "euclidean")
Diamond.HC = hclust(Diamond.Dist, method = "ward.D")
plot(Diamond.HC)

#cut trees
Diamond.HC.Tree<-cutree(Diamond.HC,k=10)
tapply(diamondsub$price,Diamond.HC.Tree,mean)
tapply(diamondsub$carat,Diamond.HC.Tree,mean)


##KMEANS CLUSTERING
KmeansClustering = kmeans(diamondNorm, centers = 10)

# Examination of results
table(KmeansClustering$cluster)
tapply(diamond$price, KmeansClustering$cluster, mean)
tapply(diamond$carat, KmeansClustering$cluster, mean)

#clear enviornment
detach("package:caret", unload=TRUE)

