# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Pharsheen Rahuman M
RegisterNumber:  212224230193
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.info
data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
![SVM For Spam Mail Detection](sam.png)
<img width="917" height="502" alt="image" src="https://github.com/user-attachments/assets/be9493eb-fd7c-4bdd-9d09-ff19bba5aabd" />
<img width="125" height="48" alt="image" src="https://github.com/user-attachments/assets/a22753f7-4947-4054-9418-4e293ca31c6d" />
<img width="602" height="221" alt="image" src="https://github.com/user-attachments/assets/226ac93c-9596-4aff-a4e6-8b07898581dc" />
<img width="237" height="38" alt="image" src="https://github.com/user-attachments/assets/75e4e995-f6dd-4424-b8ab-11ef443b5e24" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
