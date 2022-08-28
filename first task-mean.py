import numpy as nm
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
dfp = pd.read_csv('first data predict.csv')
dfm = pd.read_csv('first data.csv')

## 1- Filling with missing datas with mean method using scikit-learn SimpleImputer ##
imp_mean = sp(strategy='mean')
imp_mean = imp_mean.fit(dfm)
df_mean = pd.DataFrame(imp_mean.transform(dfm))
df_mean.columns = ['experience','test_score(out of 10)','interview_score(out of 10)','salary($)']

## 2- predict salary ##
dfp_mean = dfp
ypt_mean = df_mean['salary($)']
df_mean.drop('salary($)',axis = 1,inplace=True)
lireg.fit(df_mean,ypt_mean)
dfp_mean.drop('salary($)',axis = 1,inplace=True)
predpt_mean = lireg.predict(dfp_mean)
dfp_mean['salary($)'] = predpt_mean

## 3- round and print the starting and prediction results ##
df_mean_rounded = round(df_mean)
dfp_mean_rounded = round (dfp_mean)
print('mean fileld table :')
print(df_mean_rounded)

print('mean sallary prediction results :')
print(dfp_mean_rounded)
with open('mean sallary prediction results.txt','a' ) as f :
    df_result = dfp_mean_rounded.to_string(header = True,index = False)
    f.write('mean sallary prediction results : \n')
    f.write(df_result)
f.close()
## End of Code ;/ ##