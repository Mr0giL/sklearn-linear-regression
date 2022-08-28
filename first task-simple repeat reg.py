import numpy as nm
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
df = pd.read_csv('first data.csv')
dfp = pd.read_csv('first data predict.csv')

## 1- fill missing data using simple repeat regression method from scikit-learn Linear regression ##
### sec.1 filler ###
dft = df[(df['experience'].isnull()==False)]
train_dft = dft[dft['test_score(out of 10)'].isnull()==False]
test_dft = dft[dft['test_score(out of 10)'].isnull()==True]
yt = train_dft['test_score(out of 10)']
train_dft.drop('test_score(out of 10)',axis=1,inplace=True)
lireg.fit(train_dft,yt)
test_dft.drop('test_score(out of 10)',axis=1,inplace=True)
predt = lireg.predict(test_dft)
test_dft['test_score(out of 10)'] = predt
df.loc[6,'test_score(out of 10)'] = test_dft.loc[6,'test_score(out of 10)']
### sec.2 filler ###
test_df = df[df['experience'].isnull()==True]
train_df = df[df['experience'].isnull()==False]
train_df.drop('test_score(out of 10)',axis=1,inplace=True)
test_df.drop('test_score(out of 10)',axis=1,inplace=True)
y = train_df['experience']
train_df.drop('experience',axis=1,inplace=True)
lireg.fit(train_df,y)
test_df.drop('experience',axis=1,inplace=True)
pred = lireg.predict(test_df)
test_df['experience'] = pred
df.loc[0:1,'experience']  = test_df.loc[0:1,'experience']

## 2- predict salary ##
ypt = df['salary($)']
df.drop('salary($)',axis=1,inplace=True)
lireg.fit(df,ypt)
dfp.drop('salary($)',axis=1,inplace=True)
predpt = lireg.predict(dfp)
dfp['salary($)'] = predpt

# 3- round and print the starting and prediction results ##
df_rounded = round(df)
dfp_rounded = round(dfp)
print('one repeat regression table :')
print(df_rounded)

print('one repeat regression sallary prediction results :')
print(dfp_rounded)
with open('one repeat regression sallary prediction results.txt','a' ) as f :
    df_result = dfp_rounded.to_string(header = True,index = False)
    f.write('one repeat regression sallary prediction results : \n')
    f.write(df_result)
f.close()
## End of Code ;/ ##


