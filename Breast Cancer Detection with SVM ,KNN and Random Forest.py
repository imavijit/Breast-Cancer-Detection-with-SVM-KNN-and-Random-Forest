import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing,cross_validation,model_selection


#loading the dataset
url="http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
name=['Id','Clump Thickness','Uniform Cell Size','Uniform Cell Shape','Marginal Adhesion',
      'Signle Epithelial Size','Bare Nuclei','Bland Chromatin',
      'Normal Nucleoli','Mitoses','Class']
df=pd.read_csv(url,names=name)
df.head(5)

# Preprocess the data
df.replace('?',-99999,inplace=True)
df.drop(['Id'],1,inplace=True)
print(df.axes)
print(df.shape)

df.describe()

#plotting histogram for each variable
df.hist(figsize=(10,8))
plt.show()

#creating scatter plot matrix
scatter_matrix(df, figsize=(16,16))
plt.show()

#create datasets for training
X=np.array(df.drop(['Class'],1))
y=np.array(df['Class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)


# # KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier()

k_range=list(range(1,50))
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print(np.round(k_scores,4))


from sklearn.model_selection import GridSearchCV
k_range=list(range(1,50))
parameters=dict(n_neighbors=k_range)
clf_knn=GridSearchCV(knn,parameters,cv=10,scoring='accuracy')
print("Tuning Hyper-Parameters for accuracy" )
clf_knn.fit(X_train,y_train)
print(clf_knn.best_params_)
print(np.round(clf_knn.best_score_,3))

from sklearn.metrics import accuracy_score,classification_report
pred=clf_knn.predict(X_test)
print("Accuracy_score: ", accuracy_score(y_test,pred))
print(classification_report(y_test,pred))


# # SVM

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
clf=SVC()
Cs = [0.001, 0.01, 0.1, 1, 10,100,1000]
gammas = [0.001, 0.01, 0.1, 1,10,100,1000]
parameters=[{'kernel':['rbf'],'C':Cs,'gamma':gammas},{'kernel':['linear'],'C':Cs}]
clf_svm=GridSearchCV(clf,parameters,cv=10,scoring="accuracy")
print("Tuning Hyper-Parameters for accuracy")
clf_svm.fit(X_train,y_train)
print(clf_svm.best_params_)
print(np.round(clf_svm.best_score_,3))

predictions=clf_svm.predict(X_test)
print("Accuracy_score: ",accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))


# # Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
clf_rf=RandomForestClassifier()
param_grid = {"n_estimators": np.arange(10,100,5),
              "min_samples_split": np.arange(2,100,2),
              "criterion": ["gini", "entropy"]
              }
grid_rf=GridSearchCV(clf_rf,param_grid,cv=10,scoring='accuracy')
print("Tuning Hyper-Parameters for accuracy")
grid_rf.fit(X_train,y_train)
print(grid_rf.best_params_)
print(np.round(grid_rf.best_score_,3))

prediction=grid_rf.predict(X_test)
print("Accuracy_score: ",accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))
