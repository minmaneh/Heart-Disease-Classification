#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import json
import pickle


# In[2]:


data= pd.read_csv("heart.csv")


# In[3]:


data.head()


# # Feature Selection

# In[4]:


col_name=data.columns
feature_name=col_name.drop("target")


# In[5]:


#Feature Importance to select the most relevant features
# Feature Importance with Extra Trees Classifier
X = data.drop(['target'], axis = 1)
y = data['target']


# The score might be different given the stochastic nature of the algorithm, that is the reason of computing the average score of 100 runs

# In[6]:


a=100
scores=[]
for i in range(a):
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, y)
    score_list=model.feature_importances_
    scores.insert(i,score_list)


# In[7]:


feature_num=13
average_scores=[]
for i in range(feature_num):
    mean_score=0
    for j in range(a):
        mean_score=mean_score+scores[j][i]
    mean_score=mean_score/100
    average_scores.insert(i,mean_score)


# In[8]:


average_scores


# In[9]:


#Comparing the results in the dataframe feature_name and importance
feature_importance = pd.DataFrame(
    {'feature_name ': feature_name,
     'importance': average_scores
      })


# In[10]:


feature_importance


# In[11]:


sorted_feature_importance = feature_importance.sort_values(by=['importance'], ascending=False)


# In[12]:


sorted_feature_importance


# The top 5 features having the highest correlation with the target are **cp,ca,exang,thal** and **oldpeak**.
# 
# **cp**=chest pain type
# 
# **ca**=number of major vessels (0-3) colored by flourosopy
# 
# **exang**=exercise induced angina (1 = yes; 0 = no)
# 
# **thal**=3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# **oldpeak**=ST depression induced by exercise relative to rest

# In[13]:


data_new = data[['cp', 'ca','exang','thal','oldpeak','target']]


# # Choosing Model for prediction

# The task of predicting if the patient has a heart disease or not is a classification problem, since there are 2 possible outcomes

# The classification algorithms including **Logistic Regression**, **KNN**, **SVM**, **Random Forest** and **Naive Bayes** are implemented to find the best model for the classification.

# ## Logistic Regression

# In[14]:


#The counts of 0s and 1s
data_new["target"].value_counts()


# The data is balanced, which means there is no need to artificially balance it.

# In[15]:


X = data_new.drop(['target'], axis = 1)
y = data_new['target']


# In[16]:


X_train, X_test, y_train, y_test= train_test_split(X, y)
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred_lg = log_reg.predict(X_test)


# To measure the performance of the model the **accuracy**, **precision** and **recall** are calculated.

# In[17]:


print(accuracy_score(y_test, y_pred_lg))
print(precision_score(y_test, y_pred_lg))
print(recall_score(y_test, y_pred_lg))


# ## Random Forest Classifier

# In[18]:


rand_for=RandomForestClassifier()
rand_for.fit(X_train,y_train)
y_pred_rf=rand_for.predict(X_test)


# In[19]:


print(accuracy_score(y_test, y_pred_rf))
print(precision_score(y_test, y_pred_rf))
print(recall_score(y_test, y_pred_rf))


# ## KNN

# In[20]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)


# In[21]:


print(accuracy_score(y_test, y_pred_knn))
print(precision_score(y_test, y_pred_knn))
print(recall_score(y_test, y_pred_knn))


# ## SVM

# In[22]:


svc=svm.SVC()
svc.fit(X_train,y_train)
y_pred_svc=svc.predict(X_test)


# In[23]:


print(accuracy_score(y_test, y_pred_svc))
print(precision_score(y_test, y_pred_svc))
print(recall_score(y_test, y_pred_svc))


# ## Naive Bayes

# In[24]:


gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb=gnb.predict(X_test)


# In[25]:


print(accuracy_score(y_test, y_pred_gnb))
print(precision_score(y_test, y_pred_gnb))
print(recall_score(y_test, y_pred_gnb))


# ## Comparing the Models

# In[26]:


# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
# set height of bar
Accuracy = [0.8754863813229572, 0.9299610894941635, 0.9455252918287937, 0.9027237354085603, 0.8482490272373541]
Precision = [0.8428571428571429, 0.8992248062015504, 0.9318181818181818, 0.8652482269503546, 0.8248175182481752]
Recall = [0.921875, 0.9586776859504132, 0.9609375, 0.953125, 0.8828125]
# Set position of bar on X axis
br1 = np.arange(len(Accuracy))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
# Make the plot
plt.bar(br1, Accuracy, color ='r', width = barWidth,
        edgecolor ='grey', label ='Accuracy')
plt.bar(br2, Precision, color ='g', width = barWidth,
        edgecolor ='grey', label ='Precision')
plt.bar(br3, Recall, color ='b', width = barWidth,
        edgecolor ='grey', label ='Recall')
# Adding Xticks
plt.xlabel('Prediction Model', fontweight ='bold', fontsize = 15)
plt.ylabel('Performance', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(Accuracy))],
        ['LG', 'RFC', 'KNN', 'SVM', 'NB'])
 
plt.legend()
plt.show()


# The best model for the predictions is **KNeighborsClassifier** since it has the best performace when comparing accuracy,precision and recall. 
# 
# The final model for predicting the probabilities for the two possible conditions will use KNN.

# In[27]:


y_pred_prob_knn=knn.predict_proba(X_test)


# In[28]:


y_pred_prob_knn


# The reason for not diverse probabilities is that it estimate is simply fraction of votes among nearest neighbours. Increasing the number of neighbours will enlarge the variability, but will also decrease the performance of the model.

# In[29]:


r=X_test.iloc[1]
result = r.to_json()
p=json.loads(result)


# In[30]:


#The model 
def predict_for_instance(js):
    df = pd.DataFrame.from_dict(js,orient='index')
    df=pd.DataFrame(df.T).reindex()
    rand_for = RandomForestClassifier()
    rand_for.fit(X_train,y_train)
    y_pred_rand_for=rand_for.predict_proba(df)
    df['prediction_prob_0']=y_pred_rand_for[0][0]
    df['prediction_prob_1']=y_pred_rand_for[0][1]
    df_1=df.to_json()
    df_json=json.loads(df_1)
    return df_json


# In[31]:


l=predict_for_instance(p)
print(l)


# In[32]:


# Save the model as serialized object pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(rand_for, file)


# In[ ]:





# In[ ]:




