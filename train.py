#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # OBTENIR LE DATASET

# In[2]:


df = pd.read_csv('pointure.data')
df


# # EXPLORATION DES DONNÉES

# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.describe()


# # PRE-TRAITEMENT DES DONNÉES

# In[7]:


import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])
print(encoded_labels)
df['Genre'] = encoded_labels

df


# In[8]:


df.plot()


# In[9]:


df['Genre'].diff().hist(color='k', alpha=0.5, bins=50)


# In[10]:


df['Taille(cm)'].diff().hist(color='k', alpha=0.5, bins=50)


# In[11]:


df['Poids(kg)'].diff().hist(color='k', alpha=0.5, bins=50)


# In[12]:


df['Pointure(cm)'].diff().hist(color='k', alpha=0.5, bins=50)


# In[13]:


dfplot = pd.DataFrame(df.iloc[:, lambda dfToPredict: [0, 1, 2, 3]], columns=['Genre', 'Taille(cm)', 'Poids(kg)', 'Pointure(cm)'])
dfplot.diff().hist(color='k', alpha=0.5, bins=50)


# In[14]:


color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}
dfplot.plot.box(color=color, sym='r+')


# # MATRICE DE CORRELATION ET DE PERSON

# In[15]:


from pandas.plotting import scatter_matrix
scatter_matrix(dfplot, alpha=0.2, figsize=(6, 6), diagonal='kde')


# In[16]:


sns.pairplot(dfplot, diag_kind='kde', dropna=True)


# In[17]:


corr = dfplot.corr()
corr = corr.round(3)
f, ax = plt.subplots(figsize=(16, 12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr, cmap="YlGn", square=True, ax = ax, annot=True, linewidth = 0.1)
plt.title('Corrélation de Pearson', y=1.05, size=15)
plt.show()


# # DEFINIR LES FEATURES
# # SEPARER LE DATASET EN TRAIN ET TEST

# In[18]:


X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]


# In[19]:


from sklearn.model_selection import train_test_split

#decomposer les donnees predicteurs en training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


# In[20]:


print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # FAIRE APPRENDRE LE MODELE

# In[21]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# # EVALUATION SUR LE TRAIN

# In[22]:


y_naive_bayes1 = gnb.predict(X_train)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_train.shape[0],(y_train != y_naive_bayes1).sum()))


# In[23]:


from sklearn import metrics
accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
print("Accuracy du modele Naive Bayes predit: " + str(accuracy))


recall_score = metrics.recall_score(y_train, y_naive_bayes1)
print("recall score du modele Naive Bayes predit: " + str(recall_score))

f1_score = metrics.f1_score(y_train, y_naive_bayes1)
print("F1 score du modele Naive Bayes predit: " + str(f1_score))


# # EVALUATION SUR LE TEST

# In[24]:


y_naive_bayes2 = gnb.predict(X_test)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))

recall_score_test = metrics.recall_score(y_test, y_naive_bayes2)
print("recall score du modele Naive Bayes predit: " + str(recall_score_test))

f1_score_test = metrics.f1_score(y_test, y_naive_bayes2)
print("F1 score du modele Naive Bayes predit: " + str(f1_score_test))

accuracy_test = metrics.accuracy_score(y_test, y_naive_bayes2)
print("Accuracy du modele Naive Bayes predit: " + str(accuracy_test))


# # PREDICTION SUR UNE OBSERVATION

# In[25]:


d = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
dfToPredict = pd.DataFrame(data=d) 
dfToPredict


# In[26]:


yPredict = gnb.predict(dfToPredict)
print('La classe predite est : ', yPredict)


# # Integration de MLFlow

# In[27]:


import mlflow
import mlflow.sklearn


# In[28]:


mlflow.set_experiment(experiment_name='A57_Examen')
mlflow.set_tracking_uri("http://benmassaoud.com:5000")


# In[29]:


with mlflow.start_run():
    
    mlflow.log_metric("recall_score_test", recall_score_test)
    mlflow.log_metric("f1_score_test", f1_score_test)
    mlflow.log_metric("accuracy_test", accuracy_test)
    mlflow.sklearn.log_model(gnb, "model")


# # Export des metriques

# In[32]:


with open("metrics.txt", 'w') as outfile:
        outfile.write("recall score du modele Naive Bayes predit: " + str(recall_score_test) + "\n")
        outfile.write("F1 score du modele Naive Bayes predit: " + str(f1_score_test) + "\n")
        outfile.write("Accuracy du modele Naive Bayes predit: " + str(accuracy_test) + "\n")


# In[ ]:




