---
layout: post
title: "Minimizing Churn Rate"
description: "Through analysis of financial habits"
author: "Rishabh Pande"
coverImg: "post-bg.jpg"
---



# Minimizing Churn Rate Through Analysis of Financial Habits


### Background

Subscription products often are the main source of revenue for companies across all industries. These products can come in the form of a "one size fits all" overcompassing subscription, or in multi level memberships. Regardless of how they structure their memberships, or what industry they are in, companies almost always try to minimize customer churn (aka subscription cancellations).

To retain their customers, companies first need to identify behavioral patterns that act as catalyst in disengagement with the product. 

* Market: The target audience is the entirety of a company's subscription base. They are the ones companies want to keep. 

* Product: The subscription products that customers are already enrolled in can provide value that users may not have imagined, or that they may have forgotten


### Objective

The objective of this model is to find out which users are likely to churn, so that the company focus on re-engaging these users with the product. These efforts can be email reminders about the benefit of the product, especially focusing on features that are new or that user has shown to value


In this case study we will be working for a fintech company that provides a subscription product to its users, which allows them to manage their Bank accounts (saving account, credit cards etc.), provides them with personalized coupons, informs them about latest low APR-loans available in the market, and educates them on the best available methods to save money (like free courses on financial health etc.)





```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

dataset = pd.read_csv('churn_data.csv') # Users who were 60 days enrolled, churn in the next 30
```

## Exploratory data analysis


Now, we will do some Exploratory Data Analysis (EDA) which is an approach for data analysis that employs a variety of techniques to:

* maximize insight into a data set
* uncover underlying structure
* extract important variables
* detect outliers and anomalies
* test underlying assumptions
* develop parsimonious models
* determine optimal factor settings


```python
dataset.columns
```




    Index(['user', 'churn', 'age', 'housing', 'credit_score', 'deposits',
           'withdrawal', 'purchases_partners', 'purchases', 'cc_taken',
           'cc_recommended', 'cc_disliked', 'cc_liked', 'cc_application_begin',
           'app_downloaded', 'web_user', 'app_web_user', 'ios_user',
           'android_user', 'registered_phones', 'payment_type', 'waiting_4_loan',
           'cancelled_loan', 'received_loan', 'rejected_loan', 'zodiac_sign',
           'left_for_two_month_plus', 'left_for_one_month', 'rewards_earned',
           'reward_rate', 'is_referred'],
          dtype='object')




```python
dataset.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>churn</th>
      <th>age</th>
      <th>housing</th>
      <th>credit_score</th>
      <th>deposits</th>
      <th>withdrawal</th>
      <th>purchases_partners</th>
      <th>purchases</th>
      <th>cc_taken</th>
      <th>...</th>
      <th>waiting_4_loan</th>
      <th>cancelled_loan</th>
      <th>received_loan</th>
      <th>rejected_loan</th>
      <th>zodiac_sign</th>
      <th>left_for_two_month_plus</th>
      <th>left_for_one_month</th>
      <th>rewards_earned</th>
      <th>reward_rate</th>
      <th>is_referred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55409</td>
      <td>0</td>
      <td>37.0</td>
      <td>na</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Leo</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23547</td>
      <td>0</td>
      <td>28.0</td>
      <td>R</td>
      <td>486.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Leo</td>
      <td>0</td>
      <td>0</td>
      <td>44.0</td>
      <td>1.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58313</td>
      <td>0</td>
      <td>35.0</td>
      <td>R</td>
      <td>561.0</td>
      <td>47</td>
      <td>2</td>
      <td>86</td>
      <td>47</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Capricorn</td>
      <td>1</td>
      <td>0</td>
      <td>65.0</td>
      <td>2.17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8095</td>
      <td>0</td>
      <td>26.0</td>
      <td>R</td>
      <td>567.0</td>
      <td>26</td>
      <td>3</td>
      <td>38</td>
      <td>25</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Capricorn</td>
      <td>0</td>
      <td>0</td>
      <td>33.0</td>
      <td>1.10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61353</td>
      <td>1</td>
      <td>27.0</td>
      <td>na</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Aries</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.03</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>churn</th>
      <th>age</th>
      <th>credit_score</th>
      <th>deposits</th>
      <th>withdrawal</th>
      <th>purchases_partners</th>
      <th>purchases</th>
      <th>cc_taken</th>
      <th>cc_recommended</th>
      <th>...</th>
      <th>registered_phones</th>
      <th>waiting_4_loan</th>
      <th>cancelled_loan</th>
      <th>received_loan</th>
      <th>rejected_loan</th>
      <th>left_for_two_month_plus</th>
      <th>left_for_one_month</th>
      <th>rewards_earned</th>
      <th>reward_rate</th>
      <th>is_referred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>26996.000000</td>
      <td>18969.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>...</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
      <td>23773.000000</td>
      <td>27000.000000</td>
      <td>27000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35422.702519</td>
      <td>0.413852</td>
      <td>32.219921</td>
      <td>542.944225</td>
      <td>3.341556</td>
      <td>0.307000</td>
      <td>28.062519</td>
      <td>3.273481</td>
      <td>0.073778</td>
      <td>92.625778</td>
      <td>...</td>
      <td>0.420926</td>
      <td>0.001296</td>
      <td>0.018815</td>
      <td>0.018185</td>
      <td>0.004889</td>
      <td>0.173444</td>
      <td>0.018074</td>
      <td>29.110125</td>
      <td>0.907684</td>
      <td>0.318037</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20321.006678</td>
      <td>0.492532</td>
      <td>9.964838</td>
      <td>61.059315</td>
      <td>9.131406</td>
      <td>1.055416</td>
      <td>42.219686</td>
      <td>8.953077</td>
      <td>0.437299</td>
      <td>88.869343</td>
      <td>...</td>
      <td>0.912831</td>
      <td>0.035981</td>
      <td>0.135873</td>
      <td>0.133623</td>
      <td>0.069751</td>
      <td>0.378638</td>
      <td>0.133222</td>
      <td>21.973478</td>
      <td>0.752016</td>
      <td>0.465723</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17810.500000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>507.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>35749.000000</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>542.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.780000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>53244.250000</td>
      <td>1.000000</td>
      <td>37.000000</td>
      <td>578.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>43.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>164.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>1.530000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>69658.000000</td>
      <td>1.000000</td>
      <td>91.000000</td>
      <td>838.000000</td>
      <td>65.000000</td>
      <td>29.000000</td>
      <td>1067.000000</td>
      <td>63.000000</td>
      <td>29.000000</td>
      <td>522.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>114.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>



Few quick things to note:

* 41% of users have churned
* 32 is the average age of the users
    


```python
dataset.shape
```




    (27000, 31)




```python
dataset.dtypes
```




    user                         int64
    churn                        int64
    age                        float64
    housing                     object
    credit_score               float64
    deposits                     int64
    withdrawal                   int64
    purchases_partners           int64
    purchases                    int64
    cc_taken                     int64
    cc_recommended               int64
    cc_disliked                  int64
    cc_liked                     int64
    cc_application_begin         int64
    app_downloaded               int64
    web_user                     int64
    app_web_user                 int64
    ios_user                     int64
    android_user                 int64
    registered_phones            int64
    payment_type                object
    waiting_4_loan               int64
    cancelled_loan               int64
    received_loan                int64
    rejected_loan                int64
    zodiac_sign                 object
    left_for_two_month_plus      int64
    left_for_one_month           int64
    rewards_earned             float64
    reward_rate                float64
    is_referred                  int64
    dtype: object



** Data Cleaning: ** Next step, we will clean the data and than continue with more EDA


```python
dataset[dataset.credit_score < 300]
dataset = dataset[dataset.credit_score >= 300]
```


```python
#Check null values
dataset.isna().sum()
```




    user                          0
    churn                         0
    age                           0
    housing                       0
    credit_score                  0
    deposits                      0
    withdrawal                    0
    purchases_partners            0
    purchases                     0
    cc_taken                      0
    cc_recommended                0
    cc_disliked                   0
    cc_liked                      0
    cc_application_begin          0
    app_downloaded                0
    web_user                      0
    app_web_user                  0
    ios_user                      0
    android_user                  0
    registered_phones             0
    payment_type                  0
    waiting_4_loan                0
    cancelled_loan                0
    received_loan                 0
    rejected_loan                 0
    zodiac_sign                   0
    left_for_two_month_plus       0
    left_for_one_month            0
    rewards_earned             1190
    reward_rate                   0
    is_referred                   0
    dtype: int64



Credit score and rewards earned have significant amount of null values. We will drop them from our model.


```python
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])
```


```python
# Features Histograms
fig, ax = plt.subplots(3,3, figsize=(20, 14))
sns.distplot(dataset.age, bins = 20, ax=ax[0,0])  
sns.distplot(dataset.purchases_partners, bins = 20, ax=ax[0,1]) 
sns.distplot(dataset.app_downloaded, bins = 20, ax=ax[0,2]) 
sns.distplot(dataset.deposits, bins = 20, ax=ax[1,0]) 
sns.distplot(dataset.withdrawal, bins = 20, ax=ax[1,1]) 
sns.distplot(dataset.cc_application_begin, bins = 20, ax=ax[1,2]) 
sns.distplot(dataset.cc_recommended, bins = 20, ax=ax[2,0]) 
sns.distplot(dataset.cancelled_loan, bins = 20, ax=ax[2,1]) 
sns.distplot(dataset.reward_rate, bins = 20, ax=ax[2,2]) 
plt.show()
```


![png](output_14_0.png)


Few things to note:

* Age: Distribution is right skewed, intuitively it makes sense as older people don't use the services
* Deposit/withdrawal: Majority of people have no deposit (as the data we have is for first couple of months, and for this time period, activity could be low)


```python
## Pie Plots
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
fig = plt.figure(figsize=(15, 12))
#plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
   
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 0.9, 2.1])
plt.show()
```


![png](output_16_0.png)


Few things we notice:

* Housing: Majority of users are not owners. There's good amount of renters. Most of them are unclassified
* Payment type: Biweekly is most common
* Zodiac sign: Pretty evenly distributed, except for perhaps Capricorn

Interesting to note is features like : 'waiting_4_loan', 'cancelled_loan', 'received_loan', 'rejected_loan' and left_for_one_month'are unevenly distributed. We will try to explore more to make sure these features will be useful to build our models



```python
## Exploring Uneven Features
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
```




    0    15
    1     3
    Name: churn, dtype: int64




```python
dataset[dataset2.cancelled_loan == 1].churn.value_counts()
```




    0    194
    1    187
    Name: churn, dtype: int64




```python
dataset[dataset2.received_loan == 1].churn.value_counts()
```




    1    233
    0    162
    Name: churn, dtype: int64




```python
dataset[dataset2.rejected_loan == 1].churn.value_counts()

```




    1    64
    0    17
    Name: churn, dtype: int64




```python
dataset[dataset2.left_for_one_month == 1].churn.value_counts()
```




    1    207
    0    184
    Name: churn, dtype: int64



These are pretty balanced distribution and we do not see any strong reason that these fields are biased.

Next, we will check the correlation with Response variable


```python
## Correlation with Response Variable
dataset_corr = dataset.drop(columns = ['churn', 'user']) #drop columns

dataset_corr.corrwith(dataset.churn).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 15, rot = 45,
              grid = True)
plt.show()
```


![png](output_24_0.png)


Age is negatively correlated to the response variable churn, smaller the age - more likely for it to be 1 (or churn). 

Same with deposits and withdrawal. Smaller the deposits or withdrawal  - more likely for users to churn. This makes sense, because this means that less activity user has, more likely they will churn. 



Interestingly, 'cc_taken' is correlated with churn, meaning if user has taken a credit card, they are more likely to churn (aka they are not happy with Credit card). This will be interesting to explore further.



```python
#Correlation matrix

corr=dataset_corr.corr()

sns.set(font_scale=1.3)
plt.figure(figsize=(24, 27))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features', fontsize = '32')
plt.show()
```


![png](output_26_0.png)


Obviously, best case scenario would be every feature is independent of each other and matrix above is marked around '0' meaning that they are not linearly related. 

However, that is not the case here. As we see in the matrix, correlation between 'android user' and 'ios user' is very strong. This makes sense, as if you are an android user, you are likely not a ios user. Correlation is not exactly 1 as there are probably users who have both. We will drop one of the column as it is not bringing any new information.

Additionaly,  'app_web_user' field is 1 when 'web_user' and 'app downloaded' both are '1' (aka its a function of the two fields) which makes it not an independent variable. As we want independent variables, we will drop this field. 


```python
# Removing Correlated Fields
dataset = dataset.drop(columns = ['app_web_user'])
dataset = dataset.drop(columns = ['ios_user'])
```


```python
## Data Preparation
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])
```

### One hot encoding

We will use One hot encoding which is a simple process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.


```python
# One-Hot Encoding
dataset.housing.value_counts()
dataset.groupby('housing')['churn'].nunique().reset_index()
dataset = pd.get_dummies(dataset)
dataset.columns
dataset = dataset.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])
```


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'), dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)
```


```python
# Balancing the Training Set
y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]
```


```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2
```

## Model building


```python
# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
# Predicting Test Set
y_pred = classifier.predict(X_test)
```


```python
# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
```

    Test Data Accuracy: 0.6399



```python
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
```

    SVM Accuracy: 0.652 (+/- 0.023)



```python
# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>-0.164204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deposits</td>
      <td>0.034083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>withdrawal</td>
      <td>0.031360</td>
    </tr>
    <tr>
      <th>3</th>
      <td>purchases_partners</td>
      <td>-0.761356</td>
    </tr>
    <tr>
      <th>4</th>
      <td>purchases</td>
      <td>-0.188282</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cc_taken</td>
      <td>0.043743</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cc_recommended</td>
      <td>0.069329</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cc_disliked</td>
      <td>0.035589</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cc_liked</td>
      <td>-0.003735</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cc_application_begin</td>
      <td>0.100384</td>
    </tr>
    <tr>
      <th>10</th>
      <td>app_downloaded</td>
      <td>-0.057729</td>
    </tr>
    <tr>
      <th>11</th>
      <td>web_user</td>
      <td>0.146670</td>
    </tr>
    <tr>
      <th>12</th>
      <td>android_user</td>
      <td>-0.051212</td>
    </tr>
    <tr>
      <th>13</th>
      <td>registered_phones</td>
      <td>0.098973</td>
    </tr>
    <tr>
      <th>14</th>
      <td>waiting_4_loan</td>
      <td>-0.024642</td>
    </tr>
    <tr>
      <th>15</th>
      <td>cancelled_loan</td>
      <td>0.098474</td>
    </tr>
    <tr>
      <th>16</th>
      <td>received_loan</td>
      <td>0.102405</td>
    </tr>
    <tr>
      <th>17</th>
      <td>rejected_loan</td>
      <td>0.124107</td>
    </tr>
    <tr>
      <th>18</th>
      <td>left_for_two_month_plus</td>
      <td>0.025957</td>
    </tr>
    <tr>
      <th>19</th>
      <td>left_for_one_month</td>
      <td>0.056692</td>
    </tr>
    <tr>
      <th>20</th>
      <td>reward_rate</td>
      <td>-0.282190</td>
    </tr>
    <tr>
      <th>21</th>
      <td>is_referred</td>
      <td>0.041676</td>
    </tr>
    <tr>
      <th>22</th>
      <td>housing_O</td>
      <td>-0.038284</td>
    </tr>
    <tr>
      <th>23</th>
      <td>housing_R</td>
      <td>0.046987</td>
    </tr>
    <tr>
      <th>24</th>
      <td>payment_type_Bi-Weekly</td>
      <td>-0.064121</td>
    </tr>
    <tr>
      <th>25</th>
      <td>payment_type_Monthly</td>
      <td>-0.054434</td>
    </tr>
    <tr>
      <th>26</th>
      <td>payment_type_Semi-Monthly</td>
      <td>-0.038202</td>
    </tr>
    <tr>
      <th>27</th>
      <td>payment_type_Weekly</td>
      <td>0.043886</td>
    </tr>
    <tr>
      <th>28</th>
      <td>zodiac_sign_Aquarius</td>
      <td>0.004534</td>
    </tr>
    <tr>
      <th>29</th>
      <td>zodiac_sign_Aries</td>
      <td>0.042943</td>
    </tr>
    <tr>
      <th>30</th>
      <td>zodiac_sign_Cancer</td>
      <td>0.041694</td>
    </tr>
    <tr>
      <th>31</th>
      <td>zodiac_sign_Capricorn</td>
      <td>0.066223</td>
    </tr>
    <tr>
      <th>32</th>
      <td>zodiac_sign_Gemini</td>
      <td>0.032779</td>
    </tr>
    <tr>
      <th>33</th>
      <td>zodiac_sign_Leo</td>
      <td>0.016060</td>
    </tr>
    <tr>
      <th>34</th>
      <td>zodiac_sign_Libra</td>
      <td>0.005363</td>
    </tr>
    <tr>
      <th>35</th>
      <td>zodiac_sign_Pisces</td>
      <td>0.056116</td>
    </tr>
    <tr>
      <th>36</th>
      <td>zodiac_sign_Sagittarius</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>37</th>
      <td>zodiac_sign_Scorpio</td>
      <td>0.002033</td>
    </tr>
    <tr>
      <th>38</th>
      <td>zodiac_sign_Taurus</td>
      <td>0.013963</td>
    </tr>
    <tr>
      <th>39</th>
      <td>zodiac_sign_Virgo</td>
      <td>0.041674</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
```


```python
# Model to Test
classifier = LogisticRegression()
```


```python
# Select Best X Features
rfe = RFE(classifier, 20)
rfe = rfe.fit(X_train, y_train)
```


```python
# summarize the selection of the attributes
print(rfe.support_)
```

    [ True False False  True  True  True  True False False  True  True  True
      True  True False  True  True  True False  True  True  True  True  True
     False False False  True False False False  True False False False False
     False False False False]



```python
print(rfe.ranking_)
```

    [ 1  9  7  1  1  1  1  5 18  1  1  1  1  1 15  1  1  1 12  1  1  1  1  1
      3  2  4  1 20  8 10  1 13 16 19  6 14 21 17 11]



```python
X_train.columns[rfe.support_]
```




    Index(['age', 'purchases_partners', 'purchases', 'cc_taken', 'cc_recommended',
           'cc_application_begin', 'app_downloaded', 'web_user', 'android_user',
           'registered_phones', 'cancelled_loan', 'received_loan', 'rejected_loan',
           'left_for_one_month', 'reward_rate', 'is_referred', 'housing_O',
           'housing_R', 'payment_type_Weekly', 'zodiac_sign_Capricorn'],
          dtype='object')




```python
# New Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = X_train[X_train.columns[rfe.support_]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})    
plt.show()
```


![png](output_47_0.png)



```python
# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
# Predicting Test Set
y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])
```


```python
# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)
```




    0.6318327974276529




```python
df_cm = pd.DataFrame(cm, index = (1, 0), columns = (1, 0))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
```

    Test Data Accuracy: 0.6378



```python
 #Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

```

    SVM Accuracy: 0.651 (+/- 0.027)



```python
# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>-0.164930</td>
    </tr>
    <tr>
      <th>1</th>
      <td>purchases_partners</td>
      <td>-0.753390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>purchases</td>
      <td>-0.139449</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cc_taken</td>
      <td>0.050407</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cc_recommended</td>
      <td>0.071737</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cc_application_begin</td>
      <td>0.104003</td>
    </tr>
    <tr>
      <th>6</th>
      <td>app_downloaded</td>
      <td>-0.057866</td>
    </tr>
    <tr>
      <th>7</th>
      <td>web_user</td>
      <td>0.147560</td>
    </tr>
    <tr>
      <th>8</th>
      <td>android_user</td>
      <td>-0.051496</td>
    </tr>
    <tr>
      <th>9</th>
      <td>registered_phones</td>
      <td>0.099842</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cancelled_loan</td>
      <td>0.098161</td>
    </tr>
    <tr>
      <th>11</th>
      <td>received_loan</td>
      <td>0.101879</td>
    </tr>
    <tr>
      <th>12</th>
      <td>rejected_loan</td>
      <td>0.122173</td>
    </tr>
    <tr>
      <th>13</th>
      <td>left_for_one_month</td>
      <td>0.056976</td>
    </tr>
    <tr>
      <th>14</th>
      <td>reward_rate</td>
      <td>-0.287096</td>
    </tr>
    <tr>
      <th>15</th>
      <td>is_referred</td>
      <td>0.042179</td>
    </tr>
    <tr>
      <th>16</th>
      <td>housing_O</td>
      <td>-0.038413</td>
    </tr>
    <tr>
      <th>17</th>
      <td>housing_R</td>
      <td>0.048422</td>
    </tr>
    <tr>
      <th>18</th>
      <td>payment_type_Weekly</td>
      <td>0.089569</td>
    </tr>
    <tr>
      <th>19</th>
      <td>zodiac_sign_Capricorn</td>
      <td>0.051997</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Formatting Final Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)
final_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>churn</th>
      <th>predicted_churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20839</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15359</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34210</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57608</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11790</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1826</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8508</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>50946</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>50130</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>55422</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>259</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>17451</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>41909</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>38825</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>19314</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>26916</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>30614</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>30329</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>38853</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>15592</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40888</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>17918</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>52613</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>725</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>51797</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2601</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>33990</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>10006</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>19296</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>12135</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3763</th>
      <td>64494</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3764</th>
      <td>1185</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3765</th>
      <td>17908</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3766</th>
      <td>52426</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3767</th>
      <td>41552</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3768</th>
      <td>52762</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3769</th>
      <td>35892</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3770</th>
      <td>28025</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3771</th>
      <td>55416</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3772</th>
      <td>14997</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3773</th>
      <td>25667</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>44166</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>50893</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>10975</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3777</th>
      <td>38184</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3778</th>
      <td>31601</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3779</th>
      <td>31167</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3780</th>
      <td>51126</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3781</th>
      <td>58440</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3782</th>
      <td>65088</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3783</th>
      <td>26821</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>25599</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>3369</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>33587</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>22318</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>67681</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3789</th>
      <td>49145</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3790</th>
      <td>47206</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3791</th>
      <td>22377</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3792</th>
      <td>47663</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3793 rows × 3 columns</p>
</div>



## Conclusion

Our model has provided us with an indication of which users are likely to churn. We have purposfully left the date of the expected churn open-ended because we are focussed on only gauging the features that indicate disengagement with the product, and not the exact manner in which users will disengage. We chose this open ended emphasis to get a sense of those who are even just a bit likely to churn because we are not aiming to create new products for people who are going to leave us for sure, but for people who are starting to lose interest in the app. 

If after creating new product features , we start seeing our model predict that less of our users are going to churn, then we can assume our customers are feeling more engaged with what we are offering them. 

We can move forward with these new efforts by inquiring the opinion of users about the new features (eg. survey). If we want to transition into predicting churn more accurately , in order to put emhpasis directly on those users leaving the product, then we can add a time dimension to churn, which would add more accuracy to our model. 





