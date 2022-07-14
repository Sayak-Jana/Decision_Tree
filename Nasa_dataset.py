from matplotlib import  pyplot as   plt
import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

data=pd.read_csv("C:\\Users\\SAYAK JANA\\OneDrive\\Desktop\\my personal\\NASA\\neo.csv")





# Dropping the irrelavant feature
X = data.drop(['id','name','orbiting_body','sentry_object','hazardous'],axis=1)
#print(X.shape)

Y= LabelEncoder()
y=data['hazardous']
Y=Y.fit_transform(y)
# print(Y)



# Splitting the data in to train test
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.5)





# Fitting in to the model

model_DTC=DecisionTreeClassifier()
#print(model_DTC)
model_DTC.fit(X_train,Y_train)
prediction = model_DTC.predict(x_test)
print(prediction)
print(y_test)





# To check the accuracy
accuracy= model_DTC.score(x_test,y_test)
print(accuracy)




# Confusion matrix
#print(prediction)
#print(y_test)

confusion_model= pd.crosstab(prediction,y_test)
print(confusion_model)
print(classification_report(y_test,prediction))




























