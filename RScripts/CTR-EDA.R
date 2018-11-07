# Author: Hidehisa Arai

data1 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt1.csv"))

# Classify by age
head(data1)
data1$agecat <- cut(data1$Age, c(-Inf, 0, 18, 24, 34, 44, 54, 64, Inf))

# Print summary
summary(data1)
install.packages("doBy")
library("doBy")
siterange <- function(x) {c(length(x), min(x), max(x))}
summaryBy(Age~agecat, data=data1, FUN=siterange)
summaryBy(Gender+Signed_In+Impressions+Clicks~agecat, data=data1)

install.packages("ggplot2")
library("ggplot2")
ggplot(data1, aes(x=Impressions, fill=agecat)) + geom_histogram(binwidth=1)
ggplot(data1, aes(x=agecat, y=Impressions, fill=agecat)) + geom_boxplot()

data1$hasimps <- cut(data1$Impressions, c(-Inf, 0, Inf))
summaryBy(Clicks~hasimps, data=data1, FUN=siterange)
ggplot(subset(data1, Impressions>0), aes(x=Clicks/Impressions,
       colour=agecat)) + geom_density()
ggplot(subset(data1, Clicks>0), aes(x=Clicks/Impressions,
       colour=agecat)) + geom_density()

