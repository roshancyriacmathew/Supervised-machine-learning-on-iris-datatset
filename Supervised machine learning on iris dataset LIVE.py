#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


from sklearn.datasets import load_iris
iris_data = load_iris()


# In[3]:


print(iris_data.DESCR)


# In[4]:


df = pd.DataFrame(iris_data.data)
df.columns = iris_data.feature_names
df.head()


# In[5]:


df['Species'] = iris_data.target
df.head()


# In[6]:


df['Species'].unique()


# In[7]:


print("Species name:", iris_data['target_names'])


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


df['Species'].value_counts()


# In[13]:


sns.pairplot(df, hue='Species', markers='+', palette='colorblind')
plt.show()


# In[14]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[15]:


df.plot(kind='scatter', x="sepal length (cm)", y="sepal width (cm)")
plt.show()


# In[16]:


fig=df[df.Species==0].plot(kind='scatter', x="sepal length (cm)", y="sepal width (cm)", 
                           color='orange', label='Setosa')
df[df.Species==1].plot(kind='scatter', x="sepal length (cm)", y="sepal width (cm)", 
                           color='blue', label='Versicolor', ax=fig)
df[df.Species==2].plot(kind='scatter', x="sepal length (cm)", y="sepal width (cm)", 
                           color='green', label='Virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length vs width")
plt.show()


# In[17]:


sns.violinplot(x="Species", y='sepal length (cm)', data=df)


# In[18]:


plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
sns.violinplot(x="Species", y='sepal length (cm)', data=df)
plt.subplot(2,2,2)
sns.violinplot(x="Species", y='sepal width (cm)', data=df)
plt.subplot(2,2,3)
sns.violinplot(x="Species", y='petal length (cm)', data=df)
plt.subplot(2,2,4)
sns.violinplot(x="Species", y='petal width (cm)', data=df)
plt.show()


# In[19]:


X = df.drop('Species', axis=1)
y = df['Species']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[21]:


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# In[22]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)
dtree_acc = accuracy_score(dtree_pred, y_test)
print("Test accuracy: {:.2f}%".format(dtree_acc*100))


# In[23]:


print(classification_report(y_test, dtree_pred))


# In[24]:


print(confusion_matrix(y_test, dtree_pred))


# In[26]:


plt.figure(figsize=(15,10))
plot_tree(dtree, feature_names=iris_data.feature_names, class_names = iris_data['target_names'])


# In[27]:


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(knn_pred, y_test)
print("Test accuracy: {:.2f}%".format(knn_acc*100))


# In[28]:


print(confusion_matrix(y_test, knn_pred))


# In[29]:


df.head()


# In[30]:


df.columns


# In[31]:


data = {'sepal length (cm)':5.0, 'sepal width (cm)':3.4, 'petal length (cm)':1.4,
       'petal width (cm)':0.2}
index = [0]
new_df = pd.DataFrame(data, index)
new_df


# In[32]:


value_pred = dtree.predict(new_df)
value_pred

