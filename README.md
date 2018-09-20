
# Correlation, Regression and Multicollinearity

Here you'll practice some of the concepts from the previous lesson and begin to investigate multicollinearity in your data. **Multicollinearity is when features in your X space have correlation with each other.**  This can lead to some interesting problems in regression where it is difficult to determine precise coefficient weights for the various features. The general idea is that with the highly correlated features, you can get nearly identical performance with different combinations of those correlated features; 1 of this and 1 of that, 2 of this and none of that, 1.5 of this and .5 of that, 50 of this and -48 of that all would produce the same result if our two features are perfectly correlated.

In the following, we break down and demonstrate these concepts in more detail.

### Standard Package Imports


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

# Practice Merging

As with the previous class, practice importing and merging two datasets. The two datasets are in two files: **Walmart_Sales_Forecasting.csv** and **features.csv**. Import both of these with pandas and then merge the two into a single master dataframe. (The two columns you should merge on are date and store number.)


```python
# Your code here
# Import and merge the two files.
```

## pd.plotting.scater_matrix()
Another very useful built in pandas feature is the scatter matrix method.  
You can use this to both examine the pairwise relations between columns as well as the distribution of individual features along the diagonal. 

For example, notice how **'MarkDown1** and **MarkDown4** appear to have some substantial correlation. Also notice the histograms along the diagonal; these are the distributions of the various variables. For example, Store in the upper left is relatively uniform (we have roughly equal amounts of data for the various store numbers), and Temperature is roughly normally distributed. In contrast, most of the MarkDown features have a high degree of skew (there are a few high outliers but most of the data falls within a much smaller range).


```python
#Warning: This will take several minutes to generate! (May also freeze computers with limited specs)
pd.plotting.scatter_matrix(df.drop('IsHoliday', axis=1), figsize=(15,15));
```


![png](index_files/index_6_0.png)


## Correlation
As we can see, there's quite a bit of correlated features here!  
We can also further investigate a single relationship between two variables with the **plt.scatter(x,y)** method or calculte the pearson correlation coefficient with numpy's built in **np.corrcoeff()** method:


```python
x , y = df.MarkDown1, df.MarkDown4
print(np.corrcoef(x,y))
plt.scatter(x,y)
```

    [[nan nan]
     [nan nan]]





    <matplotlib.collections.PathCollection at 0x1a1cd16c88>




![png](index_files/index_8_2.png)



```python
temp = df[(~df.MarkDown1.isnull())
         & (~df.MarkDown4.isnull())]
x , y = temp.MarkDown1, temp.MarkDown4
print(np.corrcoef(x,y))
plt.scatter(x,y)
```

    [[1.         0.81923816]
     [0.81923816 1.        ]]





    <matplotlib.collections.PathCollection at 0x1a1cdc04e0>




![png](index_files/index_9_2.png)


## Correlation versus causation
As you may have heard before, correlation does not equal causation. One fun example of this is ice cream sales and shark attacks. We have a bizarre dataset recording ice cream sales and shark attacks on given days at various beach towns, and plotting the two we notice a distinct correlation between the two. This does not mean that more ice sales causes more shark attacks to happen. In this case, both variables (ice cream sales and shark attacks) are correlated with a third feature we have yet to examine: temperature. In summer, as the temperature rises, both ice cream sales and shark attacks increase while in winter, there are comparitively few of both. In sum, don't assume that just because two variables are correlated that there is any direct causal relation between the two.

## Multicollinearity
Multicollinearity is when we have multiple predictive variables which are highly correlated. This leads to a number of issues when we then go to perform regression (which we will investigate in more depth!) 

In our current example, MarkDown1 and MarkDown4 were highly correlated which will greatly impact our regression analysis. Let's investigate this briefly.

# Regression
Regression algorithms are designed to predict a numeric value. That could be the anticipated number of votes for a candidate, the gross sales for a product, the value of a home, or the number of yelp review for a restaurant. If you're trying to predict a numerical value, you'll want to use a regression algorithm. You may have seen simple linear regression in a previous class. This type of regression is known as ordinary least squares; it determines a line of best fit by minimizing the sum of squares of the errors between the models predictions and the actual data. In algebra and statistics classes, this is often limited to the simple 2 variable case of y=mx+b, but this process can be generalized to use multiple predictive variables. While there can be a lot of underlying mathematics, Python and SKlearn make generating and tuning these models incredibly straightforward. Here's a general outline:  
* Import packages
* Define X and y; X can be multiple columns, y is what you want to predict
    * X must have null values removed
* Initialize a regression object
* Fit a regression model
* Use your regression model to generate predictions
* Measure the accuracy of your model's predictions


```python
df.head(2)
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
      <th>Store</th>
      <th>Date</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>MarkDown1</th>
      <th>MarkDown2</th>
      <th>MarkDown3</th>
      <th>MarkDown4</th>
      <th>MarkDown5</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>IsHoliday</th>
      <th>Dept</th>
      <th>Weekly_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2010-02-05</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>211.096358</td>
      <td>8.106</td>
      <td>False</td>
      <td>1</td>
      <td>24924.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2010-02-05</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>211.096358</td>
      <td>8.106</td>
      <td>False</td>
      <td>2</td>
      <td>50605.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
           'IsHoliday', 'Dept', 'Weekly_Sales'],
          dtype='object')




```python
#1) Import packages
from sklearn.linear_model import LinearRegression

#2) Define X and y
X = df[['Store', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3',
        'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Dept']] #Only numeric features work

y = df['Weekly_Sales']

#3) Initialize a regression object
linreg = LinearRegression()

# 4) Fit the model
linreg.fit(X, y)

# 5) Use the model to predict outputs
df['Estimated_Weekly_Sales'] = linreg.predict(X)

# 6) Measure performance
# Here we print the model's R^2 to measure overall performance; the correlation between our model and the data
print('R^2 model score:', linreg.score(X,y), '\n')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-51-1c69de66056d> in <module>()
         12 
         13 # 4) Fit the model
    ---> 14 linreg.fit(X, y)
         15 
         16 # 5) Use the model to predict outputs


    ~/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/base.py in fit(self, X, y, sample_weight)
        480         n_jobs_ = self.n_jobs
        481         X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
    --> 482                          y_numeric=True, multi_output=True)
        483 
        484         if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:


    ~/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        571     X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
        572                     ensure_2d, allow_nd, ensure_min_samples,
    --> 573                     ensure_min_features, warn_on_dtype, estimator)
        574     if multi_output:
        575         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,


    ~/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        451                              % (array.ndim, estimator_name))
        452         if force_all_finite:
    --> 453             _assert_all_finite(array)
        454 
        455     shape_repr = _shape_repr(array.shape)


    ~/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py in _assert_all_finite(X)
         42             and not np.isfinite(X).all()):
         43         raise ValueError("Input contains NaN, infinity"
    ---> 44                          " or a value too large for %r." % X.dtype)
         45 
         46 


    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').


## Handling Null Values

As you can see, the model won't run with Null values in our dataset. There are many ways to handle null values including imputing methods such as using the average or median. For now, we'll simply remove all null values from X.


```python
df = df.dropna()
```

## Reruning the Modeling Process


```python
# Code from above

#1) Import packages
from sklearn.linear_model import LinearRegression

#2) Define X and y
X = df[['Store', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3',
        'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Dept']] #Only numeric features work

y = df['Weekly_Sales']

#3) Initialize a regression object
linreg = LinearRegression()

# 4) Fit the model
linreg.fit(X, y)

# 5) Use the model to predict outputs
df['Estimated_Weekly_Sales'] = linreg.predict(X)

# 6) Measure performance
# Here we print the model's R^2 to measure overall performance; the correlation between our model and the data
print('R^2 model score:', linreg.score(X,y), '\n')

#Print the coefficients for the model's formula
print('Model feature coefficients:')
weight_dict = dict(zip(X.columns, linreg.coef_))
for feat, coeff in list(zip(X.columns, linreg.coef_)):
    print(feat, 'Coefficient weight: {}'.format(round(coeff, 4)))
```

    R^2 model score: 0.034802219578679794 
    
    Model feature coefficients:
    Store Coefficient weight: -108.1617
    Temperature Coefficient weight: 38.4918
    Fuel_Price Coefficient weight: -827.9562
    MarkDown1 Coefficient weight: 0.1411
    MarkDown2 Coefficient weight: 0.0635
    MarkDown3 Coefficient weight: 0.1624
    MarkDown4 Coefficient weight: -0.0512
    MarkDown5 Coefficient weight: 0.2407
    CPI Coefficient weight: -30.2301
    Unemployment Coefficient weight: -615.576
    Dept Coefficient weight: 112.2021


# Problems with multicollinearity
There are a few considerations to keep in mind when it comes to interpreting regression models based on underlying data with multicollinearity. *One is that the coefficients in the model themselves lose interpretability.* Under ideal conditions, we would like to interpret our coefficients literally. For example, the coefficient associated with temperature is 38.49. If this coefficient were stable, we could say something along the lines "as a the temperature goes up by 1, the weekly sales goes up by 38.49. Unfortunately, with multicollinearity, we cannot make such claims. That is because the coefficients associated with weights that are highly correlated may vary widely. For example, observe what happens to the coefficients when we remove a feature with correlation to others:


```python
# Code from above

#1) Import packages
from sklearn.linear_model import LinearRegression

#2) Define X and y
#Removed MarkDown4
X = df[['Store', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3',
        'MarkDown5', 'CPI', 'Unemployment', 'Dept']] #Only numeric features work

y = df['Weekly_Sales']

#3) Initialize a regression object
linreg = LinearRegression()

# 4) Fit the model
linreg.fit(X, y)

# 5) Use the model to predict outputs
df['Estimated_Weekly_Sales'] = linreg.predict(X)

# 6) Measure performance
# Here we print the model's R^2 to measure overall performance; the correlation between our model and the data
print('R^2 model score:', linreg.score(X,y), '\n')

#Save results
new_weight_dict = dict(zip(X.columns, linreg.coef_))

#Print the coefficients for the model's formula
print('Model feature coefficients:')
for feat, coeff in list(zip(X.columns, linreg.coef_)):
    print(feat, 'Coefficient weight: {}'.format(round(coeff, 4)))
```

    R^2 model score: 0.03473682576876003 
    
    Model feature coefficients:
    Store Coefficient weight: -108.0325
    Temperature Coefficient weight: 39.7291
    Fuel_Price Coefficient weight: -692.1175
    MarkDown1 Coefficient weight: 0.1078
    MarkDown2 Coefficient weight: 0.0658
    MarkDown3 Coefficient weight: 0.1621
    MarkDown5 Coefficient weight: 0.2407
    CPI Coefficient weight: -29.8958
    Unemployment Coefficient weight: -612.1828
    Dept Coefficient weight: 112.1943


What this goes to demonstrate is that some of these coefficients are unstable and depend on what other features are incorporated into the model. Adding additional features that are correlated with features already added will never reduce the overall performance of the model, but can be thought of as 'not adding much new information'. In this way, adding correlated features is unlikely to drastically increase model performance to any substantial degree. This also makes it difficult to judge the importance of a particular variable; the importance of the variable depends on what variables are already present (if other variables already exist which are highly correlated, then the variable will again add little predictive information).


```python
weights = pd.DataFrame.from_dict(weight_dict, orient='index').reset_index()
weights.columns = ['Feature', 'Original_Weight']
weights['New_Weight'] = weights.Feature.map(new_weight_dict)
weights['Change'] = np.abs(weights['New_Weight'] - weights['Original_Weight']) #Net change (absolute value)
weights = weights.sort_values(by='Change', ascending=False)
weights['MkDwn4_Corr'] = weights.Feature.map(lambda feat: np.corrcoef(df.MarkDown4, df[feat])[0][1])
weights['Percent_Change'] = weights.Change / weights.Original_Weight
weights.sort_values(by='Percent_Change', ascending=False)
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
      <th>Feature</th>
      <th>Original_Weight</th>
      <th>New_Weight</th>
      <th>Change</th>
      <th>MkDwn4_Corr</th>
      <th>Percent_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>MarkDown1</td>
      <td>0.141131</td>
      <td>0.107823</td>
      <td>0.033308</td>
      <td>0.828928</td>
      <td>0.236005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MarkDown2</td>
      <td>0.063488</td>
      <td>0.065768</td>
      <td>0.002280</td>
      <td>-0.017517</td>
      <td>0.035913</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Temperature</td>
      <td>38.491844</td>
      <td>39.729088</td>
      <td>1.237244</td>
      <td>-0.057895</td>
      <td>0.032143</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MarkDown3</td>
      <td>0.162446</td>
      <td>0.162095</td>
      <td>0.000351</td>
      <td>-0.080216</td>
      <td>0.002158</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MarkDown5</td>
      <td>0.240696</td>
      <td>0.240669</td>
      <td>0.000027</td>
      <td>0.101114</td>
      <td>0.000114</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dept</td>
      <td>112.202109</td>
      <td>112.194276</td>
      <td>0.007833</td>
      <td>0.004650</td>
      <td>0.000070</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Store</td>
      <td>-108.161678</td>
      <td>-108.032453</td>
      <td>0.129226</td>
      <td>0.002050</td>
      <td>-0.001195</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Unemployment</td>
      <td>-615.575960</td>
      <td>-612.182834</td>
      <td>3.393126</td>
      <td>0.018817</td>
      <td>-0.005512</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CPI</td>
      <td>-30.230099</td>
      <td>-29.895812</td>
      <td>0.334287</td>
      <td>-0.041662</td>
      <td>-0.011058</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fuel_Price</td>
      <td>-827.956231</td>
      <td>-692.117515</td>
      <td>135.838715</td>
      <td>-0.025784</td>
      <td>-0.164065</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MarkDown4</td>
      <td>-0.051211</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Also notice how MarkDown1, the feature with the highest correlation to MarkDown4, which we removed, is the coefficient that changes the largest proportionally. This shows how the original coefficient weight was impacted by the highly correlated feature.

## Dummy Variables

Another huge problem with our current model is the data we have fed into it. Specifically, we passed two columns, Store and department which were numeric, but are really more of categorical variables then quantities on a discrete or continuous scale. For example, store 2 and store 1 aren't really numeric quantities, they are simply labels designating which store A or B, we are talking about. The proper way to encode this data is as binary flags (0 or 1) designating 'Is_This_Store_A?' and 'Is_This_Store_B'. This will lead to a large number of new columns, one for each possible value that a categorical variable would take on. For example, we have 45 stores in our dataset, so this will require 45 new columns, one for each store. Each of these columns will then have a binary variable (0 or 1) that will indicate whether that particular data point is associated with that particular store. While this may sound like a lot of tedious work in practice, pandas has a useful built in feature called **get_dummies()** which makes this process easy.

### Before Transformation


```python
df.Store.head()
```




    6587    1
    6588    1
    6589    1
    6590    1
    6591    1
    Name: Store, dtype: int64



### After Transformation


```python
pd.get_dummies(df.Store).head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6587</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6588</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6589</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6590</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6591</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df.head(2)
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
      <th>Store</th>
      <th>Date</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>MarkDown1</th>
      <th>MarkDown2</th>
      <th>MarkDown3</th>
      <th>MarkDown4</th>
      <th>MarkDown5</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>IsHoliday</th>
      <th>Dept</th>
      <th>Weekly_Sales</th>
      <th>Estimated_Weekly_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6587</th>
      <td>1</td>
      <td>2011-11-11</td>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>2406.62</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>False</td>
      <td>1</td>
      <td>18689.54</td>
      <td>14168.563441</td>
    </tr>
    <tr>
      <th>6588</th>
      <td>1</td>
      <td>2011-11-11</td>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>2406.62</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>False</td>
      <td>2</td>
      <td>44936.47</td>
      <td>14280.757717</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
           'IsHoliday', 'Dept', 'Weekly_Sales', 'Estimated_Weekly_Sales'],
          dtype='object')



## Similarly, we will map a numeric scale to our binary variable


```python
df.IsHoliday.value_counts()
```




    False    87064
    True      9992
    Name: IsHoliday, dtype: int64




```python
df.IsHoliday = df.IsHoliday.map({True:1, False:0})
df.IsHoliday.value_counts()
```




    0    87064
    1     9992
    Name: IsHoliday, dtype: int64



## If your categorical variables are non numeric you can do multiple at a time


```python
for col in ['Store', 'Dept']:
    df[col] = df[col].astype(str) #Make nonumeric
X = pd.get_dummies(df[['Store', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3',
        'MarkDown5', 'CPI', 'Unemployment', 'Dept', 'IsHoliday']])
X.head()
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
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>MarkDown1</th>
      <th>MarkDown2</th>
      <th>MarkDown3</th>
      <th>MarkDown5</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>IsHoliday</th>
      <th>Store_1</th>
      <th>...</th>
      <th>Dept_90</th>
      <th>Dept_91</th>
      <th>Dept_92</th>
      <th>Dept_93</th>
      <th>Dept_94</th>
      <th>Dept_95</th>
      <th>Dept_96</th>
      <th>Dept_97</th>
      <th>Dept_98</th>
      <th>Dept_99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6587</th>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6588</th>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6589</th>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6590</th>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6591</th>
      <td>59.11</td>
      <td>3.297</td>
      <td>10382.9</td>
      <td>6115.67</td>
      <td>215.07</td>
      <td>6551.42</td>
      <td>217.998085</td>
      <td>7.866</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 135 columns</p>
</div>



### Practice

Using your newly defined X above, fit a new linear regression model and print the model's r^2 score.


```python
#Your code here
```
