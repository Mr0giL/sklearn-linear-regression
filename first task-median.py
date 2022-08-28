import numpy as nm
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
dfp = pd.read_csv('first data predict.csv')
dfm = pd.read_csv('first data.csv')

## 1- Filling with missing datas with median using scikit-learn SimpleImputer ##
imp_median = sp(strategy='median')
imp_median = imp_median.fit(dfm)
df_median = pd.DataFrame(imp_median.transform(dfm))
df_median.columns = ['experience','test_score(out of 10)','interview_score(out of 10)','salary($)']

## 2- predict salary##
dfp_median = dfp
ypt_median = df_median['salary($)']
df_median.drop('salary($)',axis = 1,inplace=True)
lireg.fit(df_median,ypt_median)
dfp_median.drop('salary($)',axis = 1,inplace=True)
predpt_median = lireg.predict(dfp_median)
dfp_median['salary($)'] = predpt_median

## 3- round and print the starting and prediction results ##
df_median_rounded = round(df_median)
dfp_median_rounded = round (dfp_median)
print('median fileld table :')
print(df_median_rounded)

print('median sallary prediction results :')
print(dfp_median_rounded)
## End of Code ;/ ##


