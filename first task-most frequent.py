from cgi import test
import numpy as nm
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
dfp = pd.read_csv('first data predict.csv')
dfm = pd.read_csv('first data.csv')

## 1- Filling with missing datas with most frequent method using scikit-learn SimpleImputer ##
imp_msf = sp(strategy='most_frequent')
imp_msf = imp_msf.fit(dfm)
df_msf = pd.DataFrame(imp_msf.transform(dfm))
df_msf.columns = ['experience','test_score(out of 10)','interview_score(out of 10)','salary($)']

## 2- predict salary ##
dfp_msf = dfp
ypt_msf = df_msf['salary($)']
df_msf.drop('salary($)',axis = 1,inplace=True)
lireg.fit(df_msf,ypt_msf)
dfp_msf.drop('salary($)',axis = 1,inplace=True)
predpt_msf = lireg.predict(dfp_msf)
dfp_msf['salary($)'] = predpt_msf

## 3- round and print the starting and prediction results ##
df_msf_rounded = round(df_msf)
dfp_msf_rounded = round (dfp_msf)
print('most frequent fileld table :')
print(df_msf_rounded)

print('most frequent sallary prediction results :')
print(dfp_msf_rounded)
with open('most frequent sallary prediction results.txt','a' ) as f :
    df_result = dfp_msf_rounded.to_string(header = True,index = False)
    f.write('most frequent sallary prediction results : \n')
    f.write(df_result)
f.close()
## End of Code ;/ ##


