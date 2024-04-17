#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Ganesh Sarla , Sai Akhil Gouravarapu
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install ucimlrepo')
get_ipython().system('pip install -U imbalanced-learn')


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


# In[4]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 


# In[6]:


print(wine_quality.data.targets.describe())


# In[7]:


counts = y.value_counts()
total = counts.sum()
ratios = [count/total for count in counts]
x_values = list(range(len(counts)))

plt.bar(x_values, ratios)
plt.xticks(x_values, counts.index)
plt.xlabel('Quality Rating')  
plt.ylabel('Percentage')
plt.show()


# In[9]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import StandardScaler

# Split data
try:
  X_train, X_test, y_train, y_test = train_test_split(X, y)
except ValueError as e:
  print("Error splitting data:", e)
  raise


# In[12]:


# Scale the training and test data using StandardScaler
scaler = StandardScaler()
try:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    # Handle any scaling errors
    print("Error scaling data:", e)
    raise

# Define a K-Fold cross-validator with 10 splits
kf = KFold(n_splits=10)

# Evaluate Logistic Regression using cross-validation
log_scores = np.empty((3, 10))
for i in range(3):
    try:
        # Use cross_val_score to assess Logistic Regression performance
        scores = cross_val_score(LogisticRegression(), X_train_scaled, y_train, cv=kf)
    except Exception as e:
        # Handle any cross-validation errors
        print("Error running CV:", e)
        continue

    log_scores[i, :] = scores

# Calculate mean and standard deviation of Logistic Regression scores
log_mean = log_scores.mean(axis=0).mean()
log_std = log_scores.std(axis=0).mean()

# Print the mean and standard deviation of Logistic Regression scores
print(f'Logistic Regression: {log_mean} ± {log_std}')

# Evaluate Decision Tree using cross-validation
tree_scores = []
for _ in range(3):
    # Use cross_val_score to assess Decision Tree performance
    scores = cross_val_score(DecisionTreeClassifier(), X_train_scaled, y_train, cv=kf)
    tree_scores.append(scores)

# Convert the list of Decision Tree scores to a NumPy array
tree_scores = np.array(tree_scores)

# Print the mean and standard deviation of Decision Tree scores
print('Decision Tree: {} +/- {}'.format(tree_scores.mean(axis=0).mean(), tree_scores.std(axis=0).mean()))


# In[13]:


print('Logistic Regression: {} +/- {}'.format(log_scores.mean(), log_scores.std())) 

print('Decision Tree: {} +/- {}'.format(tree_scores.mean(), tree_scores.std()))


# In[14]:


# Assume LogisticRegression was better
final_model = LogisticRegression().fit(X_train_scaled, y_train)


# In[15]:


test_scores = final_model.score(X_test_scaled, y_test)
print('Test Accuracy: ', test_scores)


# In[16]:


y_pred = final_model.predict(X_test_scaled)


# In[17]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[18]:


import pickle
with open('wine_quality_model.pkl', 'wb') as file:
  pickle.dump(final_model, file)


# In[19]:


# Predictions on test set
y_pred = final_model.predict(X_test_scaled)

# Accuracy score
print("Accuracy: {:.4f}".format(final_model.score(X_test_scaled, y_test)))

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[20]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

print("Class ratios before oversampling:")
print(y_train.value_counts()/len(y_train))

print("Class ratios after oversampling:") 
print(y_resampled.value_counts()/len(y_resampled))


# In[21]:


# Logistic Regression Cross-Validation
kf = KFold(n_splits=10)  # Define a K-Fold cross-validator with 10 splits

log_scores = np.empty((3, 10))  # Create an empty array to store Logistic Regression scores

for i in range(3):
    # Use cross_val_score to assess Logistic Regression performance
    scores = cross_val_score(LogisticRegression(), X_resampled, y_resampled, cv=kf)
    log_scores[i, :] = scores  # Store the scores for each iteration

log_mean = log_scores.mean()   # Calculate the mean of Logistic Regression scores
log_std = log_scores.std()  # Calculate the standard deviation of Logistic Regression scores

# Print the mean and standard deviation of Logistic Regression scores
print(f'Logistic Regression: {log_mean} ± {log_std}')


# Decision Tree Cross-Validation
tree_scores = np.empty((3, 10))  # Create an empty array to store Decision Tree scores

for i in range(3):
    # Use cross_val_score to assess Decision Tree performance
    scores = cross_val_score(DecisionTreeClassifier(), X_resampled, y_resampled, cv=kf)
    tree_scores[i, :] = scores  # Store the scores for each iteration

tree_mean = tree_scores.mean()  # Calculate the mean of Decision Tree scores
tree_std = tree_scores.std()  # Calculate the standard deviation of Decision Tree scores

# Print the mean and standard deviation of Decision Tree scores
print(f'Decision Tree: {tree_mean} ± {tree_std}')


# In[22]:


# Identify best classifier
if log_mean > tree_mean:
  best_model = LogisticRegression()
else:
  best_model = DecisionTreeClassifier()

# Train final model  
final_model = best_model.fit(X_resampled, y_resampled)

# Evaluate on original test set
test_scores = final_model.score(X_test_scaled, y_test)

print(f'Test Accuracy: {test_scores}')


# In[23]:


# Predictions on test set
y_pred = final_model.predict(X_test_scaled)

# Accuracy score
print("Accuracy: {:.4f}".format(final_model.score(X_test_scaled, y_test)))

# Classification report
print(classification_report(y_test, y_pred)) 

# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# In[ ]:




