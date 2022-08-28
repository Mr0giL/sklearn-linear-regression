from email import header
import numpy as nm
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
dfp = pd.read_csv('first data predict.csv')
df_reviewed = pd.read_csv('first data-reviewed.csv')
## 1- fill missing data using simple repeat regression method from scikit-learn Linear regression ##

# to fix the missing results using regression, two cection of simple repeat method used multiple times seprrately each time on only one column new dataset stored in first data-reviewed.csv file. each repeat stored in multi reg data missing fix.xlsx file #

##  2- predict salary ##
dfp_reviewed = dfp
ypt_reviewed = df_reviewed['salary($)']
df_reviewed.drop('salary($)',axis=1,inplace=True)
lireg.fit(df_reviewed,ypt_reviewed)
dfp_reviewed.drop('salary($)',axis=1,inplace=True)
predpt_reviewed = lireg.predict(dfp_reviewed)
dfp_reviewed['salary($)'] = predpt_reviewed

## 3- round and print the starting and prediction results ##
df_reviewed_rounded = round(df_reviewed)
dfp_reviewed_rounded = round (dfp_reviewed)
print('multi repeat regression fileld table :')
print(df_reviewed_rounded)

print('multi repeat regression sallary prediction results :')
print(dfp_reviewed_rounded)
with open('multi repeat regression sallary prediction results.txt','a' ) as f :
    df_result = dfp_reviewed_rounded.to_string(header = True,index = False)
    f.write('multi repeat regression sallary prediction results : \n')
    f.write(df_result)
f.close()
## End of Code ;/ ##


