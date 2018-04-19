---
layout: post
title: "Weather in Szeged 2006-2016"
description: "Regression Analysis"
author: "Rishabh Pande"
coverImg: "post-bg.jpg"
---


### What is regression?


Regression is one way of modeling the strength and direction of the relationship between a dependent or output variable (usually represented by 'y') and one or more independent or input variables (usually represented by 'x'). It differs from correlation analysis because it allows you to predict the outcome for new input or inputs you haven’t seen yet.


What types of regression are there?


There are many different types of regression. The specific family of regressions we’ll be learning are called “generalized linear models”. The important thing for us to know is that with this family of models, we need to pick a specific type of regression. The type of regression will depend on what type of data we are trying to predict.
Linear: When you’re predicting a continuous value. (What temperature will it be today?) Logistic: When you’re predicting which category your observation is in. (Is this is a cat or a dog?)What is regression? Poisson: When you’re predicting a count value. (How many dogs will I see in the park?)


Today, we’re going to practice picking the right model for our dataset and plotting it. Let's pick up a dataset!
We are going to use the szeged-weather dataset ("Historical weather around Szeged, Hungary - from 2006 to 2016") from [Kaggle](https://www.kaggle.com/rtatman/datasets-for-regression-analysis), which contains timestamped records of weather features, some numerical, some categorical. The Regression Challenge: Day 1 proposed us to take 1 variable as target Y and 1 variable as the feature X used to predict it.


```python
# importing librarires

%matplotlib inline
import numpy as np
import pandas as pd 
from pandas import read_csv
import seaborn as sns
```


```python
#Importing dataset


dataset = read_csv('weatherHistory.csv')
dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Formatted Date</th>
      <th>Summary</th>
      <th>Precip Type</th>
      <th>Temperature (C)</th>
      <th>Apparent Temperature (C)</th>
      <th>Humidity</th>
      <th>Wind Speed (km/h)</th>
      <th>Wind Bearing (degrees)</th>
      <th>Visibility (km)</th>
      <th>Loud Cover</th>
      <th>Pressure (millibars)</th>
      <th>Daily Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006-04-01 00:00:00.000 +0200</td>
      <td>Partly Cloudy</td>
      <td>rain</td>
      <td>9.472222</td>
      <td>7.388889</td>
      <td>0.89</td>
      <td>14.1197</td>
      <td>251.0</td>
      <td>15.8263</td>
      <td>0.0</td>
      <td>1015.13</td>
      <td>Partly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006-04-01 01:00:00.000 +0200</td>
      <td>Partly Cloudy</td>
      <td>rain</td>
      <td>9.355556</td>
      <td>7.227778</td>
      <td>0.86</td>
      <td>14.2646</td>
      <td>259.0</td>
      <td>15.8263</td>
      <td>0.0</td>
      <td>1015.63</td>
      <td>Partly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2006-04-01 02:00:00.000 +0200</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>9.377778</td>
      <td>9.377778</td>
      <td>0.89</td>
      <td>3.9284</td>
      <td>204.0</td>
      <td>14.9569</td>
      <td>0.0</td>
      <td>1015.94</td>
      <td>Partly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2006-04-01 03:00:00.000 +0200</td>
      <td>Partly Cloudy</td>
      <td>rain</td>
      <td>8.288889</td>
      <td>5.944444</td>
      <td>0.83</td>
      <td>14.1036</td>
      <td>269.0</td>
      <td>15.8263</td>
      <td>0.0</td>
      <td>1016.41</td>
      <td>Partly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2006-04-01 04:00:00.000 +0200</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>8.755556</td>
      <td>6.977778</td>
      <td>0.83</td>
      <td>11.0446</td>
      <td>259.0</td>
      <td>15.8263</td>
      <td>0.0</td>
      <td>1016.51</td>
      <td>Partly cloudy throughout the day.</td>
    </tr>
  </tbody>
</table>
</div>






```python


sns.pairplot(dataset[['Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity']])

```




<img width="591" alt="screen shot 2018-04-18 at 9 27 43 pm" src="https://user-images.githubusercontent.com/34928106/38966314-a073b190-434f-11e8-964d-b99a16c06266.png">


```python

corr = dataset.drop('Loud Cover', axis=1).corr() # dropping Loud Cover because it never change
sns.heatmap(corr,  cmap="YlGnBu", square=True);

```


<img width="494" alt="screen shot 2018-04-18 at 9 31 51 pm" src="https://user-images.githubusercontent.com/34928106/38966374-f16bb7aa-434f-11e8-89b6-6c076c4f1140.png">




```python

sns.violinplot(x="Precip Type", y="Temperature (C)", data=dataset, palette="YlGnBu");

```

<img width="481" alt="screen shot 2018-04-18 at 9 32 39 pm" src="https://user-images.githubusercontent.com/34928106/38966396-0c558f8c-4350-11e8-8fa5-98298d8a8e15.png">



```python

sns.violinplot(x="Precip Type", y="Humidity", data=dataset, palette="YlGnBu");

```


<img width="461" alt="screen shot 2018-04-18 at 9 33 17 pm" src="https://user-images.githubusercontent.com/34928106/38966415-23b010c6-4350-11e8-94a6-4db57c8e0458.png">



With these plots we can choose two variables to study the relationship betweem them. In this study let's focus to explore Temperature as a function of Humidity, i.e., "how humidity influences in temperature?". The correlation plot gives us the information that they're strongly opposite related. 

In our case, Temperature is a continuous value, so we choose the Linear Regression model to tackle. If the predicted variable were Precip Type, we should use Logistic Regression, but there isn't countable variables to apply Poisson on such configuration of data.


Looking for those violinplots before, I think if we apply a linear model to just one category of Precip Type, the model may be more accurate, considering the noise from the others patterns of humidity vs temperature.



```python

sns.jointplot("Humidity", "Temperature (C)", data=dataset.where(dataset['Precip Type']=='null'), kind="hex");

```


![png](output_8_0.png)


The focus on linear regression, when you are predicting with a single feature, is to fit the best line, minimizing the square error between all the samples. This line  Ŷ   is defined as an estimative of the ground truth  Y :

Y=αX+β+ϵ
 
Ŷ =α̂ X+β̂ 
 
Where  X  is the set of samples,  α  is the inclination of the curve and  β  its intercepts. The last parameter,  ϵ  is the one that our model can't explain. As we'll see, our model  Ŷ   won't be 100% accurate and this  ϵ  is the model's residual, that you should keep in mind for next diagnostics analysis on Day 2. Remeber that ^ stands for an estimative.


```python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


ls = linear_model.LinearRegression()
```


```python
data = dataset.where(dataset['Precip Type']=='null')
data.dropna(inplace=True)
```


```python
X = data["Humidity"].values.reshape(-1,1)
y = data["Temperature (C)"].values.reshape(-1,1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
```


```python
ls.fit(X_train, y_train)
print("alpha = ",ls.coef_[0])
print("beta = ",ls.intercept_)
print("\n\nCalculating some regression quality metrics, which we'll discuss further on next notebooks")
y_pred = ls.predict(X_test)
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))
```

    alpha =  [-23.87202223]
    beta =  [ 28.30317387]
    
    
    Calculating some regression quality metrics, which we'll discuss further on next notebooks
    MSE =  9.59879754159
    R2 =  0.564908684858


Finally, we finished our regression. It means that now we have a relationship between Humidity and Temperature, where we can predict how hot/cold would be, considering we have measured the humity change. I think that was a good way to wake up the regression instincts in your data heart! Next analysis will show you how to interpret if those fits were good enough to trust in critical scenarios. Let's test it!


```python
hypothetical_humidity = 0.7
temperature_output = ls.predict(hypothetical_humidity)[0][0]
print("For such {} humidity, Linear Regression predict a temperature of {}C".format(hypothetical_humidity, 
                                                                                    round(temperature_output,1)))
```

    For such 0.7 humidity, Linear Regression predict a temperature of 11.6C

