---
layout: post
title: "Machine Learning"
description: "Example of a project"
author: "Rishabh Pande"
coverImg: "post-bg.jpg"
---

# Random Forest
### 1) Introduction

For this project we will be exploring the use of tree methods to classify schools as Private or Public based off their features.Let's start by getting the data which is included in the ISLR library, [the College data frame](https://cran.r-project.org/web/packages/ISLR/ISLR.pdf).

### 2) Load Libraries
Let's start by installing the library

```

install.packages('ISLR')
install.packages ('ggplot2')
install.packages ('caTools')
install.packages('rpart')
install.packages('rpart.plot')

```
### 3) Read the data

Let's check the head of College, which is a built in data frame with ISLR. 
```

library(ISLR)
head(College)

```

The data frame has 777 observations on the following 18 variables

* **Private** - A factor with levels No and Yes indicating private or public university
* **Apps** - Number of applications received
* **Accept** - Number of applications accepted
* **Enroll** - Number of new students enrolled
* **Top10perc** - Pct. new students from top 10% of H.S. class
* **Top25perc** - Pct. new students from top 25% of H.S. class
* **F.Undergrad** - Number of fulltime undergraduates
* **P.Undergrad** - Number of parttime undergraduates
* **Outstate** - Out-of-state tuition
* **Room.Board** - Room and board costs
* **Books** - Estimated book costs
* **Personal** - Estimated personal spending
* **PhD** - Pct. of faculty with Ph.D.’s
* **Terminal** - Pct. of faculty with terminal degree
* **S.F.Ratio** - Student/faculty ratio
* **perc.alumni** - Pct. alumni who donate
* **Expend** - Instructional expenditure per student
* **Grad.Rate** - Graduation rate

### 4) Explore the data

```

library(ggplot2)
ggplot(df,aes(Room.Board,Grad.Rate)) + geom_point(aes(color=Private))

```

<img width="925" alt="screen shot 2018-01-22 at 9 59 03 pm" src="https://user-images.githubusercontent.com/34928106/35257626-6589fc76-ffc8-11e7-8e62-48940daca7b6.png">

Let's create a histogram of full time undergrad students
```

ggplot(df,aes(F.Undergrad)) + geom_histogram(aes(fill=Private),color='black',bins=50)

```
<img width="924" alt="screen shot 2018-01-22 at 11 09 07 pm" src="https://user-images.githubusercontent.com/34928106/35257781-5703ff52-ffc9-11e7-8e5d-1803aca6a479.png">


Now let's create a histogram of Grad.Rate colored by Private
```

ggplot(df,aes(Grad.Rate)) + geom_histogram(aes(fill=Private),color='black',bins=50)

```
<img width="928" alt="screen shot 2018-01-22 at 11 12 24 pm" src="https://user-images.githubusercontent.com/34928106/35257851-c3ed5f96-ffc9-11e7-8406-8ac49acfa209.png">

It's interesting to note that there's a college with a grad rate more than 100%. Let's find out which and update the value to 100
```

subset(df,Grad.Rate > 100)

```


```
df['Cazenovia College','Grad.Rate'] <- 100
```

### 5) Train & Test data

Before we apply machine learning algorithms, we will need to split the data into training and testing sets. This enables to train an algorithm using the training data set and evaluate its accuracy on the test data set. An unrealistically low error value can arise due to overfitting if an algorithm is trained on the training data and evaluated for performance on the same data.

```

library(caTools)
set.seed(101) 
sample = sample.split(df$Private, SplitRatio = .70)
train = subset(df, sample == TRUE)
test = subset(df, sample == FALSE)

```
### 6) Decision Tree

We will create the model using rpart library to build a decision tree to predict whether or not a school is Private. 
