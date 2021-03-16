#!/usr/bin/env python
# coding: utf-8

# # Iris Classification using K-Nearest Neighbor Algorithm

# Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np


# Loading the Iris Data : From Scikit-learn dataset, we are calling the load_iris function and placing it into the iris_dataset variable

# In[2]:


from sklearn.datasets import load_iris
iris_dataset=load_iris()


# Printing the Keys of Iris Dataset

# In[3]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# Printing Description of Iris Dataset

# In[4]:


print(iris_dataset['DESCR'][:193] + "\n...")


# Printing Target Names of Iris Dataset : Lists species of the flowers we should predict

# In[5]:


print("Target_names: {}".format(iris_dataset['target_names']))


# Printing Feature Names of Iris Dataset :  : Lists description of the each features of the flower species

# In[6]:


print("Feature_names: {}".format(iris_dataset['feature_names']))


# Printing Type of Data of Iris Dataset

# In[7]:


print("Type of data: {}".format(type(iris_dataset['data'])))


# Printing Shape of Data of Iris Dataset

# In[8]:


print("Shape of data: {}".format(iris_dataset['data'].shape))


# Printing First Five Rows of the Data from Iris Dataset

# In[9]:


print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))


# Printing the Type of the target
# 

# In[10]:


print("Type of target: {}".format(type(iris_dataset['target'])))


# Printing the Shape of the target
# 

# In[11]:


print("Shape of target: {}".format(iris_dataset['target'].shape))


# Printing the target key and exploring the values

# In[12]:


print("Target:\n{}".format(iris_dataset['target']))


# Splitting the Dataset into Training and Testing Data

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0) 


# Shape of all the labels of train_test_split function

# In[14]:


print("X_train.shape(): {}".format(X_train.shape))
print("X_test.shape(): {}".format(y_train.shape))
print("y_train.shape(): {}".format(X_test.shape))
print("y_test.shape(): {}".format(y_test.shape))


# This is KNearestNeighbor machine learning model

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# Training the Model on Training Data

# In[16]:


knn.fit(X_train, y_train)


# Declaring the Model to New Data

# In[17]:


X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape: {}".format(X_new.shape))


# Printing the prediction of the model

# In[18]:


prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted Target Name: {}".format(
    iris_dataset['target_names'][prediction]))


# Evaluating the model

# In[19]:


y_pred = knn.predict(X_test)
print("Test Set Predictions:\n {}".format(y_pred))


# Printing the Test Set Score

# In[20]:


print("Test Set Score: {}".format(np.mean(y_pred == y_test)))

