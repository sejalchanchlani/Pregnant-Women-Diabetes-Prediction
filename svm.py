#Importing libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle

#Importing dataset
dataset=pd.read_csv("diabetes.csv")
X = dataset.iloc[:, [0,1,2,4,5,6,7]].values
Y = dataset.iloc[:, -1].values


#Splitting data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)

#fIT THE CLASSIFIER
#Form a classifier
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(xtrain,ytrain)
#predicting values

y_pred=classifier.predict(xtest)


#fIT THE CLASSIFIER
#Form a classifier
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X,Y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 2, 4, 5, 6, 7, 8]]))


















#see difference bw ytest and ypred
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(ytest,y_pred)
'''
#Visualizing the difference
from matplotlib.colors import ListedColormap
X_set,Y_set=xtrain,ytrain

x1,x2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
    
    plt.title("Classifier(Train set)")
    plt.xlabel("Age")
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
   ''' 
