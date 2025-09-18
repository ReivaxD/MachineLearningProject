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

