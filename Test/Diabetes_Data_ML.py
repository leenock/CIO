#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
Tasks- 
fit race
make the data of diag_1,2,3 ordinal
check out on whether something can be done for gender
do feature scaling
'''
# https://www.hindawi.com/journals/bmri/2014/781670/tab1/
# above link contains the details of the features

diabetes = pd.read_csv("C:\\Users\\Rohan\\Desktop\\diabetic_data.csv")

#Data Analysis

diabetes.replace(to_replace='?',value=np.nan,inplace=True)
diabetes['diabetesMed'].replace(to_replace=['Yes','No'],value=[1,0],inplace=True)
diabetes['change'].replace(to_replace=['Ch','No'],value=[1,0],inplace=True)
diabetes['readmitted'].replace(to_replace=['NO','>30','<30'],value=[0,1,2],inplace=True)
diabetes['max_glu_serum'].replace(to_replace=['Norm','>200','>300','None'],value=[3,4,6,3],inplace=True)
diabetes['age'].replace(to_replace=['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'],value=[5,15,25,35,45,55,65,75,85,95],inplace=True)
diabetes.replace(to_replace=['No','Down','Steady','Up'],value=[0,1,2,3],inplace=True)
diabetes.replace(to_replace=['Male','Female','Unknown/Invalid'],value=[0,1,np.nan],inplace=True)
#diabetes.replace(to_replace=['Hispanic','Caucasian','AfricanAmerican','Asian','Other'],value=[1,5,2,4,3],inplace=True)
#print(diabetes.groupby(['race','gender']).age.median())
#print(diabetes.groupby('race').age.mean())
#diabetes[['race','gender','age']] = diabetes[['race','gender','age']].interpolate()
#for i in diabetes.index:
#    diabetes.race[i]=round(diabetes.race[i])
diabetes.gender.fillna(diabetes.gender.mean(),inplace=True)
diabetes.race.fillna("Other",inplace=True)
print(diabetes.count()/101766)
print(diabetes.corr().diabetesMed)
diabetes = pd.get_dummies(diabetes,prefix=['race','readmitted'],columns=['race','readmitted'])
print(diabetes.columns)
diabetes.drop(columns=['insulin','change','diag_1','diag_2','diag_3','time_in_hospital','patient_nbr','weight','admission_source_id'],inplace=True)
diabetes.drop(columns=['medical_specialty','A1Cresult','payer_code','examide','citoglipton', 'metformin'],inplace=True)

# Drop Duplicate Data Points
diabetes.drop_duplicates(subset=diabetes.columns,keep='first',inplace=True)
y = diabetes['diabetesMed']
X = diabetes.drop(labels='diabetesMed',axis=1)

# Feature Normalization
from sklearn import preprocessing
X = preprocessing.normalize(X,axis=1)

# Division of data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#my_classifier = GradientBoostingClassifier()
my_classifier = RandomForestClassifier(n_estimators=150,random_state=42)
#my_classifier = DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)
X = pd.DataFrame(X,columns=X1.columns)
#print(X)
#feature importance
#fig = (pd.Series(my_classifier.feature_importances_, index=X.columns).nlargest(7).plot(kind='barh'))
#X.replace(to_replace=['0','1'],value=['Male','Female'])
#sns.countplot(x='age',hue='gender',data=X).set_title("Bar plot of Age with Gender(0:Male, 1:Female)")
#plt.show()

# Accuracy and F1 Score check
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
print('accuracy-\t',accuracy_score(y_test,predictions))
print("f1 score-\t",f1_score(y_test,predictions))
print("precision-\t",precision_score(y_test,predictions))
print('recall-\t',recall_score(y_test,predictions))

