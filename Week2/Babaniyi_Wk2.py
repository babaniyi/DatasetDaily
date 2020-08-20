import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import datetime as dt


df = pd.read_excel("WK_2_economics.xlsx")
df['observation_date'] = pd.to_datetime(df['observation_date'])
df['year'] = df['observation_date'].dt.year
df['month'] = df['observation_date'].dt.month
print (df['recession_probability'].describe())
df['recession_probability'] = df['recession_probability']/100
df['recession'] = np.where(df['recession_probability']>.50,1, 0)
print(df['recession'].value_counts())

df.isna().sum()
df2 = df.dropna(); df2.isna().sum()

df2.describe()

# We have 27 years of data (335 months)
df2.tail(4)

cat_df = df2.select_dtypes(['object'])
cat_df

# Some variables that are floats were classified as objects
for var in cat_df:
    df2[var] = df2[var].astype(float)
    
df2['year'] = df2['year'].astype('category')
df2['month'] = df2['month'].astype('category')

num_df = df2.select_dtypes(['int', 'float'])
cat_df = df2.select_dtypes(['category'])
cat_df.columns


['manufacturers_new_orders_durable_goods', 'industrial_production_index', 'new_one_family_houses_sold','observation_date', 'total_employment', 
        'consumer_credit', 'credit_delinquency_rate', 'unemployment_rate', 'federal_funds_rate', 'consumer_opinion', 
        'recession_probability', 'manufacturer_new_orders_consumer_goods', 'total_vehicle_sales']

num_df.corr()



'''
Graph and interpret individually: production index,
Interesting Relationships and/or visualisations: create a correlation map
1. Is there a relationship between employment rate and consumer behaviours, 
        --  How is the manufacturing industry affected by the number of people joining the workforce?
        -- What about when there's a recession
            -- variables to use: check correlation btw total employment and: consumer goods, durable goods + other vars

2. Relationship between number of houses sold and people joining the workforce : look at the effect when there's recession

2. plot unemployment index over time
3. bar chart of people joining the workforce
4. Plot recession probability over the years using df

Descriptive statistics
Median consumer credit across the years

Modelling

Part A. Recession
Recessions are notoriously difficult to predict. The difficulty comes from a lack of recessionsso this is a highly imbalanced dataset
1. Estimating future recession probabilities during the year
2. Classifying recession probabilities into recession and no-recession, use anomaly detection techniques and penalise wrong results (log loss)
3. Include interactions to improve the model and hyperparameter tuning using CV

PART B: Predicting vehicle sales

PART C: Predicting Unemployment rate

'''

sns.lineplot(x="year", y="unemployment_rate",
             hue="recession",
             data=df)

print(df['unemployment_rate'].describe())

