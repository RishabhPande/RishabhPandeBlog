---
layout: post
title: "UFC Data Analysis"
description: "Mixed Martial Arts"
author: "Rishabh Pande"
coverImg: "post-bg.jpg"
---


![](http://i.imgur.com/JtmJTKc.png)

**Background**


Mixed martial arts (MMA) is a full-contact combat sport that allows striking and grappling, both standing and on the ground, using techniques from other combat sports and martial arts. The Ultimate Fighting Championship (UFC) is an American mixed martial arts organization based in Las Vegas, Nevada and is the largest MMA promotion in the world and features the top-ranked fighters of the sport. Based in the United States, the UFC produces events worldwide[6] that showcase twelve weight divisions and abide by the Unified Rules of Mixed Martial Arts. This is a highly unpredictable sport 

**Overview**

Our dataset contains list of all UFC fights since 2013 with summed up entries of each fighter's round by round record preceding that fight. Each row represents a single fight - with each fighter's previous records summed up prior to the fight. Blank stats mean its the fighter's first fight since 2013 which is where granular data for UFC fights begins. Source of the data is Kaggle (https://www.kaggle.com/calmdownkarm/ufcdataset)


*Few things we will try to visualize:*

How's Age/Height related to the outcome?

Most popular way to win the fight?

Most popular locations in UFC?

****
**Import libraries and Load data**

Not all python capabilities are loaded to your working environment by default. We would need to import every library we are going to use. We will choose alias names to our modules for the sake of convenience (e.g. numpy --> np, pandas --> pd)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv('/Users/Rishabh/Desktop/data.csv')
df.head(2)
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
      <th>BPrev</th>
      <th>BStreak</th>
      <th>B_Age</th>
      <th>B_Height</th>
      <th>B_HomeTown</th>
      <th>B_ID</th>
      <th>B_Location</th>
      <th>B_Name</th>
      <th>B_Weight</th>
      <th>B__Round1_Grappling_Reversals_Landed</th>
      <th>...</th>
      <th>R__Round5_TIP_Ground Time</th>
      <th>R__Round5_TIP_Guard Control Time</th>
      <th>R__Round5_TIP_Half Guard Control Time</th>
      <th>R__Round5_TIP_Misc. Ground Control Time</th>
      <th>R__Round5_TIP_Mount Control Time</th>
      <th>R__Round5_TIP_Neutral Time</th>
      <th>R__Round5_TIP_Side Control Time</th>
      <th>R__Round5_TIP_Standing Time</th>
      <th>winby</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>23.0</td>
      <td>182.0</td>
      <td>Trento Italy</td>
      <td>2783</td>
      <td>Mezzocorona Italy</td>
      <td>Marvin Vettori</td>
      <td>84</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DEC</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>32.0</td>
      <td>175.0</td>
      <td>Careiro da Várzea, Amazonas Brazil</td>
      <td>2208</td>
      <td>Pharr, Texas USA</td>
      <td>Carlos Diego Ferreira</td>
      <td>70</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SUB</td>
      <td>blue</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 895 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1477 entries, 0 to 1476
    Columns: 895 entries, BPrev to winner
    dtypes: float64(873), int64(13), object(9)
    memory usage: 10.1+ MB



```python
df.describe()
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
      <th>BPrev</th>
      <th>BStreak</th>
      <th>B_Age</th>
      <th>B_Height</th>
      <th>B_ID</th>
      <th>B_Weight</th>
      <th>B__Round1_Grappling_Reversals_Landed</th>
      <th>B__Round1_Grappling_Standups_Landed</th>
      <th>B__Round1_Grappling_Submissions_Attempts</th>
      <th>B__Round1_Grappling_Takedowns_Attempts</th>
      <th>...</th>
      <th>R__Round5_TIP_Distance Time</th>
      <th>R__Round5_TIP_Ground Control Time</th>
      <th>R__Round5_TIP_Ground Time</th>
      <th>R__Round5_TIP_Guard Control Time</th>
      <th>R__Round5_TIP_Half Guard Control Time</th>
      <th>R__Round5_TIP_Misc. Ground Control Time</th>
      <th>R__Round5_TIP_Mount Control Time</th>
      <th>R__Round5_TIP_Neutral Time</th>
      <th>R__Round5_TIP_Side Control Time</th>
      <th>R__Round5_TIP_Standing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1477.000000</td>
      <td>1477.000000</td>
      <td>1474.000000</td>
      <td>1476.000000</td>
      <td>1477.000000</td>
      <td>1477.000000</td>
      <td>978.000000</td>
      <td>978.000000</td>
      <td>978.000000</td>
      <td>978.000000</td>
      <td>...</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.735274</td>
      <td>0.654705</td>
      <td>30.954545</td>
      <td>177.451220</td>
      <td>1964.633040</td>
      <td>73.804333</td>
      <td>0.036810</td>
      <td>0.896728</td>
      <td>0.431493</td>
      <td>2.986708</td>
      <td>...</td>
      <td>211.965278</td>
      <td>34.062500</td>
      <td>66.604167</td>
      <td>5.527778</td>
      <td>4.319444</td>
      <td>5.138889</td>
      <td>12.097222</td>
      <td>224.965278</td>
      <td>4.562500</td>
      <td>263.069444</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.895561</td>
      <td>1.057269</td>
      <td>4.020311</td>
      <td>8.561541</td>
      <td>666.949141</td>
      <td>14.980531</td>
      <td>0.193748</td>
      <td>1.255722</td>
      <td>0.830527</td>
      <td>3.987291</td>
      <td>...</td>
      <td>139.412374</td>
      <td>68.819742</td>
      <td>94.574736</td>
      <td>22.374419</td>
      <td>12.854023</td>
      <td>14.312013</td>
      <td>36.429320</td>
      <td>142.328509</td>
      <td>19.698681</td>
      <td>162.386212</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>152.000000</td>
      <td>129.000000</td>
      <td>52.000000</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>172.000000</td>
      <td>1755.000000</td>
      <td>65.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>110.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>126.750000</td>
      <td>0.000000</td>
      <td>139.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>177.000000</td>
      <td>2156.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>214.000000</td>
      <td>0.000000</td>
      <td>9.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>232.000000</td>
      <td>0.000000</td>
      <td>291.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>34.000000</td>
      <td>182.000000</td>
      <td>2337.000000</td>
      <td>84.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>294.500000</td>
      <td>47.500000</td>
      <td>109.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>299.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.000000</td>
      <td>7.000000</td>
      <td>46.000000</td>
      <td>213.000000</td>
      <td>2882.000000</td>
      <td>120.000000</td>
      <td>2.000000</td>
      <td>9.000000</td>
      <td>6.000000</td>
      <td>33.000000</td>
      <td>...</td>
      <td>647.000000</td>
      <td>496.000000</td>
      <td>529.000000</td>
      <td>144.000000</td>
      <td>91.000000</td>
      <td>62.000000</td>
      <td>264.000000</td>
      <td>659.000000</td>
      <td>128.000000</td>
      <td>841.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 886 columns</p>
</div>




```python
df.describe(include="all")
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
      <th>BPrev</th>
      <th>BStreak</th>
      <th>B_Age</th>
      <th>B_Height</th>
      <th>B_HomeTown</th>
      <th>B_ID</th>
      <th>B_Location</th>
      <th>B_Name</th>
      <th>B_Weight</th>
      <th>B__Round1_Grappling_Reversals_Landed</th>
      <th>...</th>
      <th>R__Round5_TIP_Ground Time</th>
      <th>R__Round5_TIP_Guard Control Time</th>
      <th>R__Round5_TIP_Half Guard Control Time</th>
      <th>R__Round5_TIP_Misc. Ground Control Time</th>
      <th>R__Round5_TIP_Mount Control Time</th>
      <th>R__Round5_TIP_Neutral Time</th>
      <th>R__Round5_TIP_Side Control Time</th>
      <th>R__Round5_TIP_Standing Time</th>
      <th>winby</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1477.000000</td>
      <td>1477.000000</td>
      <td>1474.000000</td>
      <td>1476.000000</td>
      <td>1471</td>
      <td>1477.000000</td>
      <td>1470</td>
      <td>1477</td>
      <td>1477.000000</td>
      <td>978.000000</td>
      <td>...</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>144.000000</td>
      <td>1461</td>
      <td>1477</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>568</td>
      <td>NaN</td>
      <td>431</td>
      <td>719</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Rio de Janeiro Brazil</td>
      <td>NaN</td>
      <td>Rio de Janeiro Brazil</td>
      <td>Tim Means</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DEC</td>
      <td>red</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>NaN</td>
      <td>38</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>720</td>
      <td>867</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.735274</td>
      <td>0.654705</td>
      <td>30.954545</td>
      <td>177.451220</td>
      <td>NaN</td>
      <td>1964.633040</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.804333</td>
      <td>0.036810</td>
      <td>...</td>
      <td>66.604167</td>
      <td>5.527778</td>
      <td>4.319444</td>
      <td>5.138889</td>
      <td>12.097222</td>
      <td>224.965278</td>
      <td>4.562500</td>
      <td>263.069444</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.895561</td>
      <td>1.057269</td>
      <td>4.020311</td>
      <td>8.561541</td>
      <td>NaN</td>
      <td>666.949141</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.980531</td>
      <td>0.193748</td>
      <td>...</td>
      <td>94.574736</td>
      <td>22.374419</td>
      <td>12.854023</td>
      <td>14.312013</td>
      <td>36.429320</td>
      <td>142.328509</td>
      <td>19.698681</td>
      <td>162.386212</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>152.000000</td>
      <td>NaN</td>
      <td>129.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>172.000000</td>
      <td>NaN</td>
      <td>1755.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>126.750000</td>
      <td>0.000000</td>
      <td>139.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>177.000000</td>
      <td>NaN</td>
      <td>2156.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>9.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>232.000000</td>
      <td>0.000000</td>
      <td>291.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>34.000000</td>
      <td>182.000000</td>
      <td>NaN</td>
      <td>2337.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>109.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>299.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.000000</td>
      <td>7.000000</td>
      <td>46.000000</td>
      <td>213.000000</td>
      <td>NaN</td>
      <td>2882.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>529.000000</td>
      <td>144.000000</td>
      <td>91.000000</td>
      <td>62.000000</td>
      <td>264.000000</td>
      <td>659.000000</td>
      <td>128.000000</td>
      <td>841.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 895 columns</p>
</div>




```python
print("Number of records : ", df.shape[0])
print("Number of Blue fighters : ", len(df.B_ID.unique()))
print("Number of Red fighters : ", len(df.R_ID.unique()))
```

    Number of records :  1477
    Number of Blue fighters :  715
    Number of Red fighters :  627



```python
df.isnull().sum(axis=0)
```




    BPrev                                                       0
    BStreak                                                     0
    B_Age                                                       3
    B_Height                                                    1
    B_HomeTown                                                  6
    B_ID                                                        0
    B_Location                                                  7
    B_Name                                                      0
    B_Weight                                                    0
    B__Round1_Grappling_Reversals_Landed                      499
    B__Round1_Grappling_Standups_Landed                       499
    B__Round1_Grappling_Submissions_Attempts                  499
    B__Round1_Grappling_Takedowns_Attempts                    499
    B__Round1_Grappling_Takedowns_Landed                      499
    B__Round1_Strikes_Body Significant Strikes_Attempts       499
    B__Round1_Strikes_Body Significant Strikes_Landed         499
    B__Round1_Strikes_Body Total Strikes_Attempts             499
    B__Round1_Strikes_Body Total Strikes_Landed               499
    B__Round1_Strikes_Clinch Body Strikes_Attempts            499
    B__Round1_Strikes_Clinch Body Strikes_Landed              499
    B__Round1_Strikes_Clinch Head Strikes_Attempts            499
    B__Round1_Strikes_Clinch Head Strikes_Landed              499
    B__Round1_Strikes_Clinch Leg Strikes_Attempts             499
    B__Round1_Strikes_Clinch Leg Strikes_Landed               499
    B__Round1_Strikes_Clinch Significant Kicks_Attempts       499
    B__Round1_Strikes_Clinch Significant Kicks_Landed         499
    B__Round1_Strikes_Clinch Significant Punches_Attempts     499
    B__Round1_Strikes_Clinch Significant Punches_Landed       499
    B__Round1_Strikes_Clinch Significant Strikes_Attempts     499
    B__Round1_Strikes_Clinch Significant Strikes_Landed       499
                                                             ... 
    R__Round5_Strikes_Kicks_Attempts                         1333
    R__Round5_Strikes_Kicks_Landed                           1333
    R__Round5_Strikes_Knock Down_Landed                      1333
    R__Round5_Strikes_Leg Total Strikes_Attempts             1450
    R__Round5_Strikes_Leg Total Strikes_Landed               1450
    R__Round5_Strikes_Legs Significant Strikes_Attempts      1333
    R__Round5_Strikes_Legs Significant Strikes_Landed        1333
    R__Round5_Strikes_Legs Total Strikes_Attempts            1350
    R__Round5_Strikes_Legs Total Strikes_Landed              1350
    R__Round5_Strikes_Punches_Attempts                       1333
    R__Round5_Strikes_Punches_Landed                         1333
    R__Round5_Strikes_Significant Strikes_Attempts           1333
    R__Round5_Strikes_Significant Strikes_Landed             1333
    R__Round5_Strikes_Total Strikes_Attempts                 1333
    R__Round5_Strikes_Total Strikes_Landed                   1333
    R__Round5_TIP_Back Control Time                          1333
    R__Round5_TIP_Clinch Time                                1333
    R__Round5_TIP_Control Time                               1333
    R__Round5_TIP_Distance Time                              1333
    R__Round5_TIP_Ground Control Time                        1333
    R__Round5_TIP_Ground Time                                1333
    R__Round5_TIP_Guard Control Time                         1333
    R__Round5_TIP_Half Guard Control Time                    1333
    R__Round5_TIP_Misc. Ground Control Time                  1333
    R__Round5_TIP_Mount Control Time                         1333
    R__Round5_TIP_Neutral Time                               1333
    R__Round5_TIP_Side Control Time                          1333
    R__Round5_TIP_Standing Time                              1333
    winby                                                      16
    winner                                                      0
    Length: 895, dtype: int64



**Missing values**

We oberserve there are some missing values in our data.  I know Age and Height are important features in any combat sport and they have handful of missing values. 

We will address the missing values in age and height. We can simply delete rows with missing values, but usually we would want to take advantage of as many data points as possible. Replacing missing values with zeros would not be a good idea - as age 0 will have actual meanings and that would change our data.

Therefore a good replacement value would be something that doesn't affect the data too much, such as the median or mean. the "fillna" function replaces every NaN (not a number) entry with the given input (the mean of the column in our case). Let's do this for both 'Blue' and 'Red' fighters.


```python
df['B_Age'] = df['B_Age'].fillna(np.mean(df['B_Age']))
df['B_Height'] = df['B_Height'].fillna(np.mean(df['B_Height']))
df['R_Age'] = df['R_Age'].fillna(np.mean(df['R_Age']))
df['R_Height'] = df['R_Height'].fillna(np.mean(df['R_Height']))
```

**Data Visualization **

Let's start by looking who's winning more from our dataset:


```python
#draw a bar plot 
sns.countplot(x='winner',data=df)
plt.title('Whos winning more',color = 'blue',fontsize=15)
```




    <matplotlib.text.Text at 0x113eeedd8>




![png](output_11_1.png)


Here I will just follow my instinct and play around a bit with what I feel will matter.

Let's talk about Age - a critical factor in any sport. We will start by looking at the distribution of Age from our dataset


```python
#fig, ax = plt.subplots(1,2, figsize=(12, 20))
fig, ax = plt.subplots(1,2, figsize=(15, 5))
sns.distplot(df.B_Age, ax=ax[0])
sns.distplot(df.R_Age, ax=ax[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11421ab00>




![png](output_13_1.png)


Age is a big factor in any sport, moresoever in MMA where you must have combination of strength, agility and speed (among other skills). These skills peak at 27-35 and fighter's fighting at this age should have higher likelyhood of winning the fight. Let's validate by grouping age for Blue fighters who have won the fight. 


```python
BAge = df.groupby(['B_Age']).count()['winner']
BlueAge = BAge.sort_values(axis=0, ascending=False)
BlueAge.head(10)
```




    B_Age
    30.0    164
    33.0    138
    29.0    134
    32.0    128
    27.0    120
    31.0    112
    28.0    106
    34.0    106
    26.0     72
    35.0     67
    Name: winner, dtype: int64



Clearly, most fights have been won by fighters in their late 20’s through early 30’s as they peak during this time and then lose strength, quickness and cardiovascular capacity

On the other hand, younger fighters do not develop peak strength till 27-28~ while older fighters are usually slower and more likely to lose. Let's check if this is true in our data. This time we will check for 'Red' fighters. 


```python
RAge = df.groupby(['R_Age']).count()['winner']
RedAge = RAge.sort_values(axis=0, ascending=False)
RedAge.tail(10)
```




    R_Age
    24.000000    25
    23.000000    17
    40.000000    10
    41.000000    10
    22.000000    10
    21.000000     5
    43.000000     4
    44.000000     3
    46.000000     2
    31.380081     1
    Name: winner, dtype: int64



Looks like this is true. It makes me curious about the total number of Red and Blue fighters who are younger than 35.  


```python
fig, ax = plt.subplots(1,2, figsize=(15, 5))
above35 =['above35' if i >= 35 else 'below35' for i in df.B_Age]
df_B = pd.DataFrame({'B_Age':above35})
sns.countplot(x=df_B.B_Age, ax=ax[0])
plt.ylabel('Number of fighters')
plt.title('Age of Blue fighters',color = 'blue',fontsize=15)

above35 =['above35' if i >= 35 else 'below35' for i in df.R_Age]
df_R = pd.DataFrame({'R_Age':above35})
sns.countplot(x=df_R.R_Age, ax=ax[1])
plt.ylabel('Number of Red fighters')
plt.title('Age of Red fighters',color = 'Red',fontsize=15)
```




    <matplotlib.text.Text at 0x11782dd30>




![png](output_19_1.png)


Interestingly, most fighters are below 35. MMA is a brutal sport for older guys and can leave them with lifelong injuries. 

Lastly, let's look at the mean difference


```python
df['Age_Difference'] = df.B_Age - df.R_Age
df[['Age_Difference', 'winner']].groupby('winner').mean()
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
      <th>Age_Difference</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>blue</th>
      <td>-1.459711</td>
    </tr>
    <tr>
      <th>draw</th>
      <td>-1.555556</td>
    </tr>
    <tr>
      <th>no contest</th>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>red</th>
      <td>0.273304</td>
    </tr>
  </tbody>
</table>
</div>



Age matters, and youth is a clear advantage. 

Height is also a major advantage in MMA as it means more the height more is the reach, meaning - taller fighter can attack from a distance keeping themselves safe from the hitting zone. Let's start by looking at the distribution of height:


```python
fig, ax = plt.subplots(1,2, figsize=(15, 5))
sns.distplot(df.B_Height, bins = 20, ax=ax[0]) #Blue 
sns.distplot(df.R_Height, bins = 20, ax=ax[1]) #Red
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1179007b8>




![png](output_23_1.png)



```python
fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(df.B_Height, shade=True, color='indianred', label='Red')
sns.kdeplot(df.R_Height, shade=True, label='Blue')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x113f762b0>




![png](output_24_1.png)



```python
df['Height Difference'] = df.B_Height - df.R_Height
df[['Height Difference', 'winner']].groupby('winner').mean()
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
      <th>Height Difference</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>blue</th>
      <td>0.118151</td>
    </tr>
    <tr>
      <th>draw</th>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>no contest</th>
      <td>-1.411765</td>
    </tr>
    <tr>
      <th>red</th>
      <td>-0.052536</td>
    </tr>
  </tbody>
</table>
</div>



Taller fighter has an advantage and, on average, wins. Of course, unless you are Rocky fighting Drago ;)

Now, let's talk about how the fighters are winning. The three most popular ways to win in an MMA fight are:

**1. DEC:**  Decision (Dec)  is a result of the fight or bout that does not end in a knockout in which the judges' scorecards are consulted to determine the winner; a majority of judges must agree on a result. A fight can either end in a win for an athlete, a draw, or a no decision.

**2. SUB: ** also referred to as a "tap out" or "tapping out" - is often performed by visibly tapping the floor or the opponent with the hand or in some cases with the foot, to signal the opponent and/or the referee of the submission

**3. KO/TKO:**  Knockout (KO) is when a fighter gets knocked out cold. (i.e.. From a standing to not standing position from receiving a strike.). Technical Knockout (TKO) is when a fighter is getting pummeled and is unable to defend him/herself further. The referee will step in and make a judgement call to end it and prevent the fighter from receiving any more unnecessary or permanent damage, and call it a TKO.


```python
sns.countplot(x='winby',data=df)
plt.title('Most popular way to win?',color = 'blue',fontsize=15)
```




    <matplotlib.text.Text at 0x117c4f860>




![png](output_27_1.png)


So most fights are going to the judges. Second most popular way is Knockout and the Technical KO.

MMA is a complex sport, in a sense it is the only sport where defense and offense could be done in the same movement. Hitting someone is a risk as it leaves you open for your opponent to counter. However, the bigger the risk, the greater the reward. More offensive attempts you make should mean more you land on your opponent (and with right skills and power - more chance you have to win the fight).
Let's see if this is true with our data.


```python
sns.lmplot(x="B__Round1_Strikes_Body Significant Strikes_Attempts", 
               y="B__Round1_Strikes_Body Significant Strikes_Landed", 
               col="winner", hue="winner", data=df, col_wrap=2, size=6)
```




    <seaborn.axisgrid.FacetGrid at 0x117c96b00>




![png](output_29_1.png)


Attempts and strikes landed are, as expected, perfectly linear.

Now, let's look at the location and find out most popular countries


```python
#Adding 2 columns to make one column

Bloc = df.groupby(['B_Location']).count()['B_ID']
location = Bloc.sort_values(axis=0, ascending=False)
location.head(10)
```




    B_Location
    Rio de Janeiro Brazil                  38
    Denver, Colorado USA                   27
    Albuquerque, New Mexico USA            25
    Coconut Creek, Florida USA             21
    Sacramento, California USA             20
    San Diego, California United States    19
    Glendale, Arizona USA                  17
    Montreal, Quebec Canada                16
    Las Vegas, Nevada USA                  16
    Coconut Creek, FL USA                  15
    Name: B_ID, dtype: int64




```python
#Adding 2 columns to make one column
Rloc = df.groupby(['R_Location']).count()['R_ID']
R_location = Rloc.sort_values(axis=0, ascending=False)
R_location.head(10)
```




    R_Location
    Rio de Janeiro Brazil                    67
    Montreal, Quebec Canada                  30
    Coconut Creek, Florida USA               29
    Denver, Colorado USA                     29
    Coconut Creek, Florida United States     29
    Las Vegas, Nevada USA                    24
    Sao Paulo Brazil                         22
    Albuquerque, New Mexico United States    21
    Dublin Ireland                           19
    Albuquerque, New Mexico USA              18
    Name: R_ID, dtype: int64



Brazil, USA and Canada are the most popular locations for UFC. 
