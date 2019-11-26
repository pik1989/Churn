# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:07:31 2019

@author: beingdatum.com
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
%matplotlib inline
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('latest.csv')

data.head(5)
data.shape

# Plot
data.CURRENT_MTH_CHURN.str.get_dummies().sum().plot.pie(label='CHURN', autopct='%1.0000f%%')

data.info()

#Converting tenure to years
data['CURRENT_MTH_CHURN'].replace(['Y','N'],[1,0],inplace=True)
data['TENUREINYEARS'] = data['TENURE']/12
data['TENUREINYEARS'] = data['TENUREINYEARS'].astype(int)




def diff_bar(x,y):
    
    data.groupby([x,y]).size().unstack(level=-1).plot(kind='bar', figsize=(10,5))
    plt.xlabel(x,fontsize= 10)
    plt.ylabel('count',fontsize= 10)
    plt.legend(loc=0,fontsize= 10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("{X} Vs {Y}".format(X=x,Y=y),fontsize = 20)
    plt.show()

diff_bar('TENUREINYEARS','CURRENT_MTH_CHURN')
diff_bar('DISTRICT','CURRENT_MTH_CHURN')
diff_bar('COSTCENTRE','CURRENT_MTH_CHURN')
diff_bar('LINE_STAT','CURRENT_MTH_CHURN')
diff_bar('BILL_CYCL','CURRENT_MTH_CHURN')


#Deleting unwanted columns
data.pop('TENURE')
data.pop('LINE_STAT') #Dropped as per Timothy, there's no relevancy of this column
data.pop('BILL_CYCL') #As there's just one bill cycle data
data.pop('IMAGE')
data.pop('EFFC_STRT_DATE')
data.pop('EFFC_EXP_DATE')
data.pop('CHURN_REASON')

data['COSTCENTRE'].replace(['SCFD','SCFY'],[1,2],inplace=True)

#Filling CHURN GROUP
data["CHURN_GROUP"] = data.CHURN_GROUP.fillna("Not Churned")

data['ACCT_NO'], data['CONT_NO'], data['LINE_NO'] = data['NEWACCT_NO'].str.split('.', 2).str

data.pop('NEWACCT_NO')

number= LabelEncoder()
data["DISTRICTNO"] = number.fit_transform(data["DISTRICT"].astype('str'))
data["CHURN_GROUP_NO"] = number.fit_transform(data["CHURN_GROUP"].astype('str'))

old = data

data.pop('DISTRICT')
data.pop('CHURN_GROUP')


data['LINE_NO_NEW'] = data['LINE_NO'].str[-5:]

data.pop('LINE_NO')

corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap="YlGnBu", annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.25)
 
print (data.isna().any(axis=1))
train_y = train['CURRENT_MTH_CHURN']
test_y = test['CURRENT_MTH_CHURN']
 
train_x = train
train_x.pop('CURRENT_MTH_CHURN')
test_x = test
test_x.pop('CURRENT_MTH_CHURN')




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
 
logisticRegr = LogisticRegression()
logisticRegr.fit(X=train_x, y=train_y)
 
test_y_pred = logisticRegr.predict(test_x)
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print('Intercept: ' + str(logisticRegr.intercept_))
print('Regression: ' + str(logisticRegr.coef_))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
print(classification_report(test_y, test_y_pred))
 
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'churn'), ('No churn', 'churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)

data['CURRENT_MTH_CHURN'].value_counts()

####################################################################
####################################################################
####################################################################
################        DOWN SAMPLING        #######################
####################################################################
####################################################################
####################################################################

from sklearn.utils import resample

 
data_majority = data[data['CURRENT_MTH_CHURN']==0]
data_minority = data[data['CURRENT_MTH_CHURN']==1]
data_majority_downsample = resample(data_majority, 
                                 replace=True,     
                                 n_samples=len(data.loc[data['CURRENT_MTH_CHURN'] == 1]),    
                                 random_state=123) 
data_minority_downsample = resample(data_minority, 
                                 replace=True,     
                                 n_samples=len(data.loc[data['CURRENT_MTH_CHURN'] == 1]),    
                                 random_state=123) 
df_train = pd.concat([data_majority_downsample, data_minority_downsample])

 
# Display new class counts
print (df_train.CURRENT_MTH_CHURN.value_counts())

train, test = train_test_split(df_train, test_size = 0.25)
 
train_y_upsampled = train['CURRENT_MTH_CHURN']
test_y_upsampled = test['CURRENT_MTH_CHURN']
 
train_x_upsampled = train
train_x_upsampled.pop('CURRENT_MTH_CHURN')
test_x_upsampled = test
test_x_upsampled.pop('CURRENT_MTH_CHURN')
 
logisticRegr_balanced = LogisticRegression()
logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)
 
test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))
print(classification_report(test_y_upsampled, test_y_pred_balanced))

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix = confusion_matrix(test_y_upsampled, test_y_pred_balanced)
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'churn'), ('No churn', 'churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)



####################################################################
####################################################################
####################################################################
################        UP SAMPLING        #########################
####################################################################
####################################################################
####################################################################

df_minority_upsampled = resample(data_minority, 
                                 replace=True,     
                                 n_samples=len(data.loc[data['CURRENT_MTH_CHURN'] == 0]),    
                                 random_state=123) 

df_train = pd.concat([data_majority, df_minority_upsampled])

 
# Display new class counts
print (df_train.CURRENT_MTH_CHURN.value_counts())

# Display new class counts
print (df_train.CURRENT_MTH_CHURN.value_counts())

train, test = train_test_split(df_train, test_size = 0.25)
 
train_y_upsampled = train['CURRENT_MTH_CHURN']
test_y_upsampled = test['CURRENT_MTH_CHURN']
 
train_x_upsampled = train
train_x_upsampled.pop('CURRENT_MTH_CHURN')
test_x_upsampled = test
test_x_upsampled.pop('CURRENT_MTH_CHURN')
 
logisticRegr_balanced = LogisticRegression()
logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)
 
test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))
print(classification_report(test_y_upsampled, test_y_pred_balanced))

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix = confusion_matrix(test_y_upsampled, test_y_pred_balanced)
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'churn'), ('No churn', 'churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
