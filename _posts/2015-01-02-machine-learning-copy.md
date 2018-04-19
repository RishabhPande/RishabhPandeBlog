---
layout: post
title: "Machine Learning"
description: "Example of a project"
author: "Rishabh Pande"
coverImg: "post-bg.jpg"
---

# New York city taxi trip duration
### 1) Introduction

This is a comprehensive Exploratory Data Analysis for the New York City Taxi Trip Duration
competition with tidy R and ggplot2.

The goal of this playground challenge is to predict the duration of taxi rides in NYC 
based on features like trip coordinates or pickup date and time. The data comes in the 
shape of 1.5 million training observations and 630k test observation.
Each row contains one taxi trip.

In this project, we will first study and visualise the original data, engineer new 
features, and examine potential outliers. Then we add two external data sets on the 
NYC weather and on the theoretically fastest routes. We visualise and analyse the new 
features within these data sets and their impact on the target trip_duration values. 
Finally, we will make a brief excursion into viewing this challenge as a classification 
problem and finish this notebook with a simple XGBoost model that provides a basic 
prediction (final part under construction).

### 2) Load Libraries
Let's start by installing the library

```

library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('alluvial') # visualisation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('lubridate') # date and time
library('geosphere') # geospatial locations
library('leaflet') # maps
library('leaflet.extras') # maps
library('maps') # maps
library('xgboost') # modelling
library('caret') # modelling

```

We will use R Cookbooks to create multi-panel plots


### 3) Read the data

We use data.table’s fread function to speed up reading in the data:

```
train <- as.tibble(fread('../input/nyc-taxi-trip-duration/train.csv'))
test <- as.tibble(fread('../input/nyc-taxi-trip-duration/test.csv'))
sample_submit <- as.tibble(fread('../input/nyc-taxi-trip-duration/sample_submission.csv'))
```

Let’s have an overview of the data sets using the summary and glimpse tools

```
summary(train)
glimpse(train)
```

We find:

vendor_id only takes the values 1 or 2, presumably to differentiate two taxi companies

pickup_datetime and (in the training set) dropoff_datetime are combinations of date and 
time that we will have to re-format into a more useful shape

passenger_count takes a median of 1 and a maximum of 9 in both data sets

The pickup/dropoff_longitute/latitute describes the geographical coordinates where the 
meter was activate/deactivated.

store_and_fwd_flag is a flag that indicates whether the trip data was sent immediately 
to the vendor (“N”) or held in the memory of the taxi because there was no connection to 
the server (“Y”). Maybe there could be a correlation with certain geographical areas with 
bad reception?

trip_duration: our target feature in the training data is measured in seconds.

### 5) Missing values

Knowing about missing values is important because they indicate how much we don’t know 
about our data. Making inferences based on just a few cases is often unwise. In addition,
many modelling procedures break down when missing values are involved and the 
corresponding rows will either have to be removed completely or the values need to be 
estimated somehow.

Here, we are in the fortunate position that our data is complete and there are no 
missing values:

```
sum(is.na(train))
## [1] 0


sum(is.na(test))
## [1] 0
```
### 6) Combining train and test

In preparation for our eventual modelling analysis we combine the train and test data 
sets into a single one. I find it generally best not to examine the test data too 
closely, since this bears the risk of overfitting your analysis to this data. However, 
a few simple consistency checks between the two data sets can be of advantage.

```
combine <- bind_rows(train %>% mutate(dset = "train"), 
                     test %>% mutate(dset = "test",
                                     dropoff_datetime = NA,
                                     trip_duration = NA))
combine <- combine %>% mutate(dset = factor(dset))
```

### 7) Reformating features

For our following analysis, we will turn the data and time from characters into date 
objects. We also recode vendor_id as a factor. This makes it easier to visualise 
relationships that involve these features.

```
train <- train %>%
  mutate(pickup_datetime = ymd_hms(pickup_datetime),
         dropoff_datetime = ymd_hms(dropoff_datetime),
         vendor_id = factor(vendor_id),
         passenger_count = factor(passenger_count))
```







