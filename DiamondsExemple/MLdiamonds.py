# Copyright (c) 2025 ReivaxD
# Tous droits réservés.
#
# Ce code est protégé par le droit d’auteur.
# Licence : usage personnel uniquement.

# Librairies :
import statsmodels.api as sm
import numpy as np
import pandas as pd
from patsy import dmatrices
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn import datasets

np.random.seed(0)

# Extraction of the database
df = pd.read_csv('DiamondsExemple/diamonds.csv', index_col=0)

# Add a column with noise in order to experiment with variable selection
df['noise'] = np.random.normal(0, 1, len(df))
df['x_noise'] = df['x'] + np.random.normal(0, 1, len(df))

"""
# If we check our columns with :

print(df.shape)
print(df.dtypes)
print(df.isna().sum())

# We can see that there are no missing values,
# We can also see that there are columns with the type "Object"
"""

# Object type could create bugs especially for the onehotencoder 
# So it's better anyways to convert them to "category"
df = df.astype({'cut': 'category', 'color': 'category', 'clarity': 'category'})

"""
If we study the plot of the data with the scatterplot 

sns.scatterplot(data=df, x='x', y='price')
sns.scatterplot(data=df, x='y', y='price')
sns.scatterplot(data=df, x='z', y='price')
plt.show()

We can observ that there are outliners values :
    -> some values for x, y and z are 0 but a lenght of 0 isn't relevant
    -> others values are very high (>= 30) but dont tally with price. 
        We could suppose that it come from  encodingmistake.

"""

# Remove outliers
for col in ['x', 'y', 'z']:
  df = df[(df[col] != 0) & (df[col] < 30)]

# Now that the dataset is clean. We could look for fitting a simple linear regression
# We use the variable "x" as predictor (and the price as output)
y, X = dmatrices('price ~ x', data=df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()

"""
The result of res.summary() is :

                            OLS Regression Results               

==============================================================================
Dep. Variable:                  price   R-squared:               
        0.787
Model:                            OLS   Adj. R-squared:          
        0.787
Method:                 Least Squares   F-statistic:             
    1.994e+05
Date:                Thu, 18 Sep 2025   Prob (F-statistic):               0.00
Time:                        18:47:17   Log-Likelihood:          
  -4.8181e+05
No. Observations:               53917   AIC:                     
    9.636e+05
Df Residuals:                   53915   BIC:                     
    9.636e+05
Df Model:                           1                            

Covariance Type:            nonrobust                            

==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept  -1.418e+04     41.329   -343.149      0.000   -1.43e+04   -1.41e+04
x           3160.2067      7.077    446.541      0.000    3146.336    3174.078
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            34564.825
Skew:                           1.190   Prob(JB):                         0.00
Kurtosis:                       6.118   Cond. No.                         31.3
==============================================================================
"""

# As we can see the model fit by our date is y = -1.418e+04 + 3160.2067*x
# With a R^2 = 0.787
# A p-value < 0.05 -> coefficients statistically significant

# If We use the variable "cut" as predictor (and the price as output)
y, X = dmatrices('price ~ cut', data=df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()


'''
The result of res.summary() is :

                            OLS Regression Results
==============================================================================
Dep. Variable:                  price   R-squared:                       0.013
Model:                            OLS   Adj. R-squared:                  0.013
Method:                 Least Squares   F-statistic:                     174.5
Date:                Fri, 19 Sep 2025   Prob (F-statistic):          9.49e-149
Time:                        14:54:41   Log-Likelihood:            -5.2318e+05
No. Observations:               53917   AIC:                         1.046e+06
Df Residuals:                   53912   BIC:                         1.046e+06
Df Model:                           4
Covariance Type:            nonrobust
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept         4357.5009     98.768     44.119      0.000    4163.915    4551.086
cut[T.Good]       -431.0974    113.829     -3.787      0.000    -654.203    -207.992
cut[T.Ideal]      -900.4956    102.389     -8.795      0.000   -1101.179    -699.812
cut[T.Premium]     221.6298    104.375      2.123      0.034      17.054     426.206
cut[T.Very Good]  -375.6703    105.140     -3.573      0.000    -581.745    -169.595
==============================================================================
Omnibus:                    15109.104   Durbin-Watson:                   0.033
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            34452.011
Skew:                           1.615   Prob(JB):                         0.00
Kurtosis:                       5.214   Cond. No.                         14.9
==============================================================================
'''

# As we can see the model fit by our date is y = 4357.5009 - 431.0974*cut[T.Good] - 900.4956*cut[T.Ideal] + 221.6298*cut[T.Premium] - 375.6703*cut[T.Very Good]
# With a R^2 = 0.013 (dropped compared to the precedent model)
# All p-value < 0.05 -> coefficients statistically significant

# However if we check differences between distribution of cat with 
# sns.boxenplot(data=df, y='price', x='cut')
# We can see that the differences are not significant enough to predict the price based only on the cut of diamonds

# As we can't use the "cut" variable alone We will see the result using "cut" and "x"
y, X = dmatrices('price ~ cut * x', data=df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()

print(res.summary())
'''
The result of res.summary() is :

                            OLS Regression Results
==============================================================================
Dep. Variable:                  price   R-squared:                       0.793
Model:                            OLS   Adj. R-squared:                  0.793
Method:                 Least Squares   F-statistic:                 2.288e+04
Date:                Fri, 19 Sep 2025   Prob (F-statistic):               0.00
Time:                        15:43:57   Log-Likelihood:            -4.8112e+05
No. Observations:               53917   AIC:                         9.623e+05
Df Residuals:                   53907   BIC:                         9.623e+05
Df Model:                           9
Covariance Type:            nonrobust
======================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------
Intercept          -1.542e+04    300.821    -51.244      0.000    -1.6e+04   -1.48e+04
cut[T.Good]         1449.6722    334.642      4.332      0.000     793.771    2105.573
cut[T.Ideal]        1168.7318    307.826      3.797      0.000     565.391    1772.073
cut[T.Premium]       546.5541    311.142      1.757      0.079     -63.286    1156.395
cut[T.Very Good]    1227.9382    313.384      3.918      0.000     613.704    1842.173
x                   3163.2553     47.577     66.487      0.000    3070.004    3256.507
cut[T.Good]:x        -99.6954     53.609     -1.860      0.063    -204.770       5.379
cut[T.Ideal]:x        50.9512     48.980      1.040      0.298     -45.051     146.953
cut[T.Premium]:x      92.3357     49.334      1.872      0.061      -4.360     189.031
cut[T.Very Good]:x     1.4355     49.894      0.029      0.977     -96.357      99.228
==============================================================================
Omnibus:                    12348.559   Durbin-Watson:                   0.439
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            35242.434
Skew:                           1.207   Prob(JB):                         0.00
Kurtosis:                       6.140   Cond. No.                         589.
==============================================================================

'''

# The model here is is y = -1.542e+04 + 1449.6722*cut[T.Good] + 1168.7318*cut[T.Ideal] + 546.5541*cut[T.Premium] + 1227.9382*cut[T.Very Good]
#                          + 3163.2553*x - 99.6954*cut[T.Good]:x + 50.9512*cut[T.Ideal]:x + 92.3357*cut[T.Premium]:x + 1.4355*cut[T.Very Good]:x
# The model seems way more complicated but it's not that mush.
# As an exemple, if we want to interpret the model for a "Premium" cut diamound. The model became :
# y = -1.542e+04 + 546.5541 + 3163.2553*x + 92.3357*x (values at T.Premium are 1 and the other are 0)

