---
layout: post
title: "UFC data analysis"
description: "Mixed martial arts"
author: "Rishabh Pande"
coverImg: "post-bg.jpg"
---

# UFC EDA


![](http://i.imgur.com/JtmJTKc.png)

- <a href='#1'>1. Introduction</a>  
- <a href='#2'>2. Loading libraries and retrieving data</a>
- <a href='#3'>3. Understanding the Data</a>
- <a href='#4'>4. Missing Values</a>
- <a href='#5'>5.  Data Visualization</a>



# <a id='1'>1. Introduction</a>

**Background**

Mixed martial arts (MMA) is a full-contact combat sport that allows striking and grappling, both standing and on the ground, using techniques from other combat sports and martial arts. The Ultimate Fighting Championship (UFC) is an American mixed martial arts organization based in Las Vegas, Nevada and is the largest MMA promotion in the world and features the top-ranked fighters of the sport. Based in the United States, the UFC produces events worldwide that showcase twelve weight divisions and abide by the Unified Rules of Mixed Martial Arts. This is a highly unpredictable sport 

Few things we will try to visualize:

* How's Age/Height related to the outcome?
* Most popular locations in UFC?
* Most popular way to win the fight?
* Comparing techniques used by fighters

****
# <a id='2'>2. Loading libraries and retrieving data</a>

Not all python capabilities are loaded to your working environment by default. We would need to import every library we are going to use. We will choose alias names to our modules for the sake of convenience (e.g. numpy --> np, pandas --> pd)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
%matplotlib inline

from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline


import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("data.csv")
df.head(2)
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.



<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>





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
      <td>0</td>
      <td>0</td>
      <td>38.0</td>
      <td>193.0</td>
      <td>Hounslow England</td>
      <td>808</td>
      <td>Amsterdam The Netherlands</td>
      <td>Alistair Overeem</td>
      <td>120.0</td>
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
      <td>blue</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>36.0</td>
      <td>172.0</td>
      <td>Chicago, Illinois United States</td>
      <td>1054</td>
      <td>Chicago, Illinois United States</td>
      <td>Ricardo Lamas</td>
      <td>65.0</td>
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
  </tbody>
</table>
<p>2 rows × 894 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2318 entries, 0 to 2317
    Columns: 894 entries, BPrev to winner
    dtypes: float64(876), int64(9), object(9)
    memory usage: 15.8+ MB


# <a id='3'>3. Understanding the data</a>

Dataset contains list of all UFC fights since 2013 with summed up entries of each fighter's round by round record preceding that fight. Created in the attempt to predict a UFC fight winner .  Each row represents a single fight - with each fighter's previous records summed up prior to the fight. Blank stats mean its the fighter's first fight since 2013 which is where granular data for UFC fights. 

We have about 895 columns, few important columns to note:

* BPrev: Previous fights by 'Blue' fighter
* B_Age: Age of 'Blue' fighter
* B_Height: Height of 'Blue' fighter
* B_Weight: Weight of 'Blue' fighter
* B_Location: Location of 'Blue' fighter
* B_Hometown: Hometown of 'Blue fighter
* RPrev: Previous fights by 'Red' fighter
* R_Age: Age of 'Red' fighter
* R_Height: Height of 'Red' fighter
* R_Weight: Weight of 'Red' fighter
* R_Location: Location of 'Red' fighter
* R_Hometown: Hometown of 'Red fighter
* Date: Date of the fight
* winby: How did the fighter win the fight (decision, submission KO etc.)
* winner: Who was the winner of the fight?

Apart from this, dataset contains all the techniques (punch, kicks, takedowns etc.) attempted and landed by the fighters in each round. 


```python
df.describe()
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
      <td>2318.000000</td>
      <td>2318.000000</td>
      <td>2301.000000</td>
      <td>2301.000000</td>
      <td>2318.000000</td>
      <td>2306.000000</td>
      <td>1647.000000</td>
      <td>1647.000000</td>
      <td>1647.000000</td>
      <td>1647.000000</td>
      <td>...</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.391286</td>
      <td>0.744607</td>
      <td>31.776184</td>
      <td>177.327249</td>
      <td>2120.001726</td>
      <td>73.699480</td>
      <td>0.074681</td>
      <td>1.103825</td>
      <td>0.577413</td>
      <td>3.852459</td>
      <td>...</td>
      <td>251.411111</td>
      <td>33.644444</td>
      <td>64.903704</td>
      <td>6.855556</td>
      <td>3.977778</td>
      <td>4.303704</td>
      <td>12.011111</td>
      <td>262.740741</td>
      <td>4.381481</td>
      <td>303.103704</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.539978</td>
      <td>1.145596</td>
      <td>4.165267</td>
      <td>8.807620</td>
      <td>705.089725</td>
      <td>15.425347</td>
      <td>0.305691</td>
      <td>1.537946</td>
      <td>1.049758</td>
      <td>5.261864</td>
      <td>...</td>
      <td>197.588561</td>
      <td>75.756504</td>
      <td>100.516057</td>
      <td>25.786323</td>
      <td>13.039241</td>
      <td>12.810181</td>
      <td>39.171198</td>
      <td>199.940529</td>
      <td>20.941919</td>
      <td>224.848078</td>
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
      <td>29.000000</td>
      <td>172.000000</td>
      <td>1910.250000</td>
      <td>61.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>137.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>137.000000</td>
      <td>0.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>177.000000</td>
      <td>2230.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>222.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>258.500000</td>
      <td>0.000000</td>
      <td>296.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>35.000000</td>
      <td>182.000000</td>
      <td>2709.000000</td>
      <td>84.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>299.000000</td>
      <td>34.000000</td>
      <td>98.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>300.000000</td>
      <td>0.000000</td>
      <td>364.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.000000</td>
      <td>10.000000</td>
      <td>48.000000</td>
      <td>213.000000</td>
      <td>3196.000000</td>
      <td>120.000000</td>
      <td>3.000000</td>
      <td>13.000000</td>
      <td>8.000000</td>
      <td>47.000000</td>
      <td>...</td>
      <td>1259.000000</td>
      <td>633.000000</td>
      <td>666.000000</td>
      <td>144.000000</td>
      <td>91.000000</td>
      <td>62.000000</td>
      <td>273.000000</td>
      <td>1291.000000</td>
      <td>200.000000</td>
      <td>1473.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 885 columns</p>
</div>




```python
df.describe(include="all")
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
      <td>2318.000000</td>
      <td>2318.000000</td>
      <td>2301.000000</td>
      <td>2301.000000</td>
      <td>2301</td>
      <td>2318.000000</td>
      <td>2305</td>
      <td>2318</td>
      <td>2306.000000</td>
      <td>1647.000000</td>
      <td>...</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>270.000000</td>
      <td>2282</td>
      <td>2318</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>724</td>
      <td>NaN</td>
      <td>567</td>
      <td>949</td>
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
      <td>Kevin Lee</td>
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
      <td>46</td>
      <td>NaN</td>
      <td>58</td>
      <td>11</td>
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
      <td>1111</td>
      <td>1327</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.391286</td>
      <td>0.744607</td>
      <td>31.776184</td>
      <td>177.327249</td>
      <td>NaN</td>
      <td>2120.001726</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.699480</td>
      <td>0.074681</td>
      <td>...</td>
      <td>64.903704</td>
      <td>6.855556</td>
      <td>3.977778</td>
      <td>4.303704</td>
      <td>12.011111</td>
      <td>262.740741</td>
      <td>4.381481</td>
      <td>303.103704</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.539978</td>
      <td>1.145596</td>
      <td>4.165267</td>
      <td>8.807620</td>
      <td>NaN</td>
      <td>705.089725</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.425347</td>
      <td>0.305691</td>
      <td>...</td>
      <td>100.516057</td>
      <td>25.786323</td>
      <td>13.039241</td>
      <td>12.810181</td>
      <td>39.171198</td>
      <td>199.940529</td>
      <td>20.941919</td>
      <td>224.848078</td>
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
      <td>29.000000</td>
      <td>172.000000</td>
      <td>NaN</td>
      <td>1910.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>137.000000</td>
      <td>0.000000</td>
      <td>178.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>177.000000</td>
      <td>NaN</td>
      <td>2230.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>258.500000</td>
      <td>0.000000</td>
      <td>296.500000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>35.000000</td>
      <td>182.000000</td>
      <td>NaN</td>
      <td>2709.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>98.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>300.000000</td>
      <td>0.000000</td>
      <td>364.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.000000</td>
      <td>10.000000</td>
      <td>48.000000</td>
      <td>213.000000</td>
      <td>NaN</td>
      <td>3196.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>666.000000</td>
      <td>144.000000</td>
      <td>91.000000</td>
      <td>62.000000</td>
      <td>273.000000</td>
      <td>1291.000000</td>
      <td>200.000000</td>
      <td>1473.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 894 columns</p>
</div>




```python
print("Number of records : ", df.shape[0])
print("Number of Blue fighters : ", len(df.B_ID.unique()))
print("Number of Red fighters : ", len(df.R_ID.unique()))
```

    Number of records :  2318
    Number of Blue fighters :  942
    Number of Red fighters :  835



```python
df.isnull().sum(axis=0)
```




    BPrev                                                       0
    BStreak                                                     0
    B_Age                                                      17
    B_Height                                                   17
    B_HomeTown                                                 17
    B_ID                                                        0
    B_Location                                                 13
    B_Name                                                      0
    B_Weight                                                   12
    B__Round1_Grappling_Reversals_Landed                      671
    B__Round1_Grappling_Standups_Landed                       671
    B__Round1_Grappling_Submissions_Attempts                  671
    B__Round1_Grappling_Takedowns_Attempts                    671
    B__Round1_Grappling_Takedowns_Landed                      671
    B__Round1_Strikes_Body Significant Strikes_Attempts       671
    B__Round1_Strikes_Body Significant Strikes_Landed         671
    B__Round1_Strikes_Body Total Strikes_Attempts             671
    B__Round1_Strikes_Body Total Strikes_Landed               671
    B__Round1_Strikes_Clinch Body Strikes_Attempts            671
    B__Round1_Strikes_Clinch Body Strikes_Landed              671
    B__Round1_Strikes_Clinch Head Strikes_Attempts            671
    B__Round1_Strikes_Clinch Head Strikes_Landed              671
    B__Round1_Strikes_Clinch Leg Strikes_Attempts             671
    B__Round1_Strikes_Clinch Leg Strikes_Landed               671
    B__Round1_Strikes_Clinch Significant Kicks_Attempts       671
    B__Round1_Strikes_Clinch Significant Kicks_Landed         671
    B__Round1_Strikes_Clinch Significant Punches_Attempts     671
    B__Round1_Strikes_Clinch Significant Punches_Landed       671
    B__Round1_Strikes_Clinch Significant Strikes_Attempts     671
    B__Round1_Strikes_Clinch Significant Strikes_Landed       671
                                                             ... 
    R__Round5_Strikes_Kicks_Attempts                         2048
    R__Round5_Strikes_Kicks_Landed                           2048
    R__Round5_Strikes_Knock Down_Landed                      2048
    R__Round5_Strikes_Leg Total Strikes_Attempts             2280
    R__Round5_Strikes_Leg Total Strikes_Landed               2280
    R__Round5_Strikes_Legs Significant Strikes_Attempts      2048
    R__Round5_Strikes_Legs Significant Strikes_Landed        2048
    R__Round5_Strikes_Legs Total Strikes_Attempts            2071
    R__Round5_Strikes_Legs Total Strikes_Landed              2071
    R__Round5_Strikes_Punches_Attempts                       2048
    R__Round5_Strikes_Punches_Landed                         2048
    R__Round5_Strikes_Significant Strikes_Attempts           2048
    R__Round5_Strikes_Significant Strikes_Landed             2048
    R__Round5_Strikes_Total Strikes_Attempts                 2048
    R__Round5_Strikes_Total Strikes_Landed                   2048
    R__Round5_TIP_Back Control Time                          2048
    R__Round5_TIP_Clinch Time                                2048
    R__Round5_TIP_Control Time                               2048
    R__Round5_TIP_Distance Time                              2048
    R__Round5_TIP_Ground Control Time                        2048
    R__Round5_TIP_Ground Time                                2048
    R__Round5_TIP_Guard Control Time                         2048
    R__Round5_TIP_Half Guard Control Time                    2048
    R__Round5_TIP_Misc. Ground Control Time                  2048
    R__Round5_TIP_Mount Control Time                         2048
    R__Round5_TIP_Neutral Time                               2048
    R__Round5_TIP_Side Control Time                          2048
    R__Round5_TIP_Standing Time                              2048
    winby                                                      36
    winner                                                      0
    Length: 894, dtype: int64




# <a id='4'>4. Missing Values</a>

We oberserve there are some missing values in our data.  I know Age and Height are important features in any combat sport and they have handful of missing values. 

We will address the missing values in age and height. We can simply delete rows with missing values, but usually we would want to take advantage of as many data points as possible. Replacing missing values with zeros would not be a good idea - as age 0 will have actual meanings and that would change our data.

Therefore a good replacement value would be something that doesn't affect the data too much, such as the median or mean. the "fillna" function replaces every NaN (not a number) entry with the given input (the mean of the column in our case). Let's do this for both 'Blue' and 'Red' fighters.


```python
df['B_Age'] = df['B_Age'].fillna(np.mean(df['B_Age']))
df['B_Height'] = df['B_Height'].fillna(np.mean(df['B_Height']))
df['R_Age'] = df['R_Age'].fillna(np.mean(df['R_Age']))
df['R_Height'] = df['R_Height'].fillna(np.mean(df['R_Height']))
```

# <a id='5'>5. Data Visualization</a>

Let's start by looking who's winning more from our dataset:


```python
temp = df["winner"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Winner",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Whos winning more",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
```


<div id="70324620-3148-4dba-bae9-1ff174cfc349" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("70324620-3148-4dba-bae9-1ff174cfc349", [{"values": [1327, 951, 24, 16], "labels": ["red", "blue", "no contest", "draw"], "domain": {"x": [0, 1]}, "hole": 0.6, "type": "pie"}], {"title": "Winner", "annotations": [{"font": {"size": 17}, "showarrow": false, "text": "Whos winning more", "x": 0.5, "y": 0.5}]}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


Here I will just follow my instinct and play around a bit with what I feel will matter.

Let's talk about Age - a critical factor in any sport. We will start by looking at the distribution of Age from our dataset


```python
#fig, ax = plt.subplots(1,2, figsize=(12, 20))
fig, ax = plt.subplots(1,2, figsize=(15, 5))
sns.distplot(df.B_Age, ax=ax[0])
sns.distplot(df.R_Age, ax=ax[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a107f12b0>




![png](output_14_1.png)


Age is a big factor in any sport, moresoever in MMA where you must have combination of strength, agility and speed (among other skills). These skills peak at 27-35 and fighter's fighting at this age should have higher likelyhood of winning the fight. Let's validate by grouping age for Blue fighters who have won the fight. 


```python
BAge = df.groupby(['B_Age']).count()['winner']
BlueAge = BAge.sort_values(axis=0, ascending=False)
BlueAge.head(10)
```




    B_Age
    31.0    245
    32.0    224
    29.0    206
    30.0    187
    35.0    173
    34.0    171
    28.0    158
    33.0    156
    27.0    138
    36.0    106
    Name: winner, dtype: int64



Clearly, most fights have been won by fighters in their late 20’s through early 30’s as they peak during this time and then lose strength, quickness and cardiovascular capacity

On the other hand, younger fighters do not develop peak strength till 27-28~ while older fighters are usually slower and more likely to lose. Let's check if this is true in our data. This time we will check for 'Red' fighters. 


```python
RAge = df.groupby(['R_Age']).count()['winner']
RedAge = RAge.sort_values(axis=0, ascending=False)
RedAge.tail(10)
```




    R_Age
    43.0    12
    22.0    11
    23.0    10
    42.0     8
    44.0     5
    46.0     3
    48.0     2
    20.0     2
    45.0     1
    21.0     1
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




    <matplotlib.text.Text at 0x1a11d253c8>




![png](output_20_1.png)


Interestingly, most fighters are below 35. MMA is a brutal sport for older guys and can leave them with lifelong injuries. 

Lastly, let's look at the mean difference


```python
df['Age_Difference'] = df.B_Age - df.R_Age
df[['Age_Difference', 'winner']].groupby('winner').mean()
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
      <td>-1.789845</td>
    </tr>
    <tr>
      <th>draw</th>
      <td>-1.187500</td>
    </tr>
    <tr>
      <th>no contest</th>
      <td>-1.893579</td>
    </tr>
    <tr>
      <th>red</th>
      <td>0.161410</td>
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




    <matplotlib.axes._subplots.AxesSubplot at 0x1a11447550>




![png](output_24_1.png)



```python
fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(df.B_Height, shade=True, color='indianred', label='Red')
sns.kdeplot(df.R_Height, shade=True, label='Blue')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a11523c18>




![png](output_25_1.png)



```python
df['Height Difference'] = df.B_Height - df.R_Height
df[['Height Difference', 'winner']].groupby('winner').mean()
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
      <td>0.304305</td>
    </tr>
    <tr>
      <th>draw</th>
      <td>2.437500</td>
    </tr>
    <tr>
      <th>no contest</th>
      <td>-0.172334</td>
    </tr>
    <tr>
      <th>red</th>
      <td>0.089707</td>
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
temp = df["winby"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      #"name": "Types of Loans",
      #"hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"How the fighter's are winning?",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Win by",
                "x": 0.50,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
```


<div id="ec5b1f10-b0e2-4fd9-9865-205f29634029" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("ec5b1f10-b0e2-4fd9-9865-205f29634029", [{"values": [1111, 744, 427], "labels": ["DEC", "KO/TKO", "SUB"], "domain": {"x": [0, 1]}, "hole": 0.6, "type": "pie"}], {"title": "How the fighter's are winning?", "annotations": [{"font": {"size": 20}, "showarrow": false, "text": "Win by", "x": 0.5, "y": 0.5}]}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


So most fights are going to the judges. Second most popular way is Knockout and the Technical KO.

Let's check how this is distibuted with respect to Age for 'Red' fighters.


```python
g = sns.FacetGrid(df, col='winby')
g.map(plt.hist, 'R_Age', bins=20)
```




    <seaborn.axisgrid.FacetGrid at 0x1a116b2390>




![png](output_30_1.png)


MMA is a complex sport, in a sense it is the only sport where defense and offense could be done in the same movement. Hitting someone is a risk as it leaves you open for your opponent to counter. However, the *bigger the risk, the greater the reward*. More offensive attempts you make should mean more you land on your opponent (and with right skills and power - more chance you have to win the fight). 

Let's see if this is true with our data.


```python
sns.lmplot(x="B__Round1_Strikes_Body Significant Strikes_Attempts", 
               y="B__Round1_Strikes_Body Significant Strikes_Landed", 
               col="winner", hue="winner", data=df, col_wrap=2, size=6)
```




    <seaborn.axisgrid.FacetGrid at 0x1a11686908>




![png](output_32_1.png)


Attempts and strikes landed are, as expected, perfectly linear.

Now, let's look at the location and find out most popular countries


```python
cnt_srs = df['R_Location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for Red fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")
```


<div id="40aee38a-e28b-4204-874e-1a0962ab649e" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("40aee38a-e28b-4204-874e-1a0962ab649e", [{"type": "bar", "x": ["Rio de Janeiro Brazil", "Albuquerque, New Mexico United States", "Coconut Creek, Florida United States", "Sao Paulo Brazil", "Montreal, Quebec Canada", "Las Vegas, Nevada United States", "Chicago, Illinois United States", "San Diego, California United States", "Denver, Colorado United States", "Miami, Florida United States", "Dublin Ireland", "Boca Raton, Florida United States", "Tokyo Japan", "Fort Worth, Texas United States", "Curitiba Brazil"], "y": [99, 59, 47, 42, 40, 40, 27, 27, 24, 22, 22, 22, 20, 18, 18], "marker": {"color": [99, 59, 47, 42, 40, 40, 27, 27, 24, 22, 22, 22, 20, 18, 18]}}], {"title": "Most Popular cities for Red fighters"}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
cnt_srs = df['B_Location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for Blue fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")
```


<div id="17e15854-6c4d-4437-9630-5a869b38e9d1" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("17e15854-6c4d-4437-9630-5a869b38e9d1", [{"type": "bar", "x": ["Rio de Janeiro Brazil", "Coconut Creek, Florida United States", "Albuquerque, New Mexico United States", "Las Vegas, Nevada United States", "San Diego, California United States", "Denver, Colorado United States", "Glendale, Arizona United States", "Chicago, Illinois United States", "Montreal, Quebec Canada", "Milwaukee, Wisconsin United States", "Sacramento, California United States", "Tokyo Japan", "Boca Raton, Florida United States", "Sao Paulo Brazil", "Miami, Florida United States"], "y": [58, 46, 43, 39, 35, 34, 29, 25, 25, 23, 22, 21, 20, 19, 19], "marker": {"color": [58, 46, 43, 39, 35, 34, 29, 25, 25, 23, 22, 21, 20, 19, 19]}}], {"title": "Most Popular cities for Blue fighters"}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


MMA seems to be most prominent in Brazil and USA. Infact, MMA is second most popular sport after Soccer in Brazil. I wonder if it is due to ancient Brazilian Jiu-Jitsu?

Now, let's look at the Grappling reversals, grappling standups and grappling takedowns landed in different weight categories in** Round 1**


```python
r1 = df[['B_Weight', 'B__Round1_Grappling_Reversals_Landed', 'B__Round1_Grappling_Standups_Landed', 
    'B__Round1_Grappling_Takedowns_Landed']].groupby('B_Weight').sum()

r1.plot(kind='line', figsize=(14,6))
plt.show()
```


![png](output_37_0.png)


There are very few Grappling reversals but high amount of Grappling takedowns that were landed. More specifically weight classes between 70 - 80 prefer takedowns during Round 1. 

Let's compare the same for Round 5


```python
r5 = df[['B_Weight', 'B__Round5_Grappling_Reversals_Landed', 'B__Round5_Grappling_Standups_Landed', 'B__Round5_Grappling_Takedowns_Landed']].groupby('B_Weight').sum()

r5.plot(kind='line', figsize=(14,6))
plt.show()
```


![png](output_39_0.png)


Interestingly, grappling reversals increase for fighters between weight 80-90, while takedowns have decreased in the lighter weight groups.

Lets look similar data for Clinch head strikes, Clinch leg strikes and Body strikes for Round 1


```python
clin_r1 = df[['B_Weight', 'B__Round1_Strikes_Clinch Head Strikes_Landed', 'B__Round1_Strikes_Clinch Leg Strikes_Landed', 'B__Round1_Strikes_Clinch Body Strikes_Landed']].groupby('B_Weight').sum()

clin_r1.plot(kind='line', figsize=(14,6))
plt.show()
```


![png](output_41_0.png)


Fighters prefer to land  more head strikes during round 1, let's compare this with what happens in Round 5:


```python
clin_r5= df[['B_Weight', 'B__Round1_Strikes_Clinch Head Strikes_Landed', 'B__Round5_Strikes_Clinch Leg Strikes_Landed', 'B__Round5_Strikes_Clinch Body Strikes_Landed']].groupby('B_Weight').sum()

clin_r5.plot(kind='line', figsize=(14,6))
plt.show()
```


![png](output_43_0.png)


By Round 5, fighters (who are now worn-out)  are hardly landing any leg and body strike. They are still landing good amount of Head strikes. This makes sense as the fight is coming to an end and instead of depending on the judges, they want to go for a Knock out. 

*More to come! **Stay tuned!***
