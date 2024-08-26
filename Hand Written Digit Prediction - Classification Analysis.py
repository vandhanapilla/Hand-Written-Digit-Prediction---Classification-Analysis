#!/usr/bin/env python
# coding: utf-8

# # Hand Written Digit Prediction - Classification Analysis

# To develop a machine learning model that can accurately classify handwritten digits from images into their corresponding digit classes (0-9).The objective of this task is to visualize and understand the structure of the digits dataset, which consists of 8x8 pixel grayscale images of handwritten digits. By examining the first few images in the dataset

# # Import Library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier


# # Import Data

# In[2]:


from sklearn.datasets import load_digits


# # Describe Data

# In[3]:


# Load the dataset
digits = load_digits()
images = digits.images
target_names = digits.target_names


# # Data Visualization 

# In[4]:


# Create a figure with subplots
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax, image, label in zip(axes, images[:4], digits.target[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Target: {target_names[label]}')
plt.show()


# In[5]:


digits.images.shape


# In[6]:


digits.images[0]


# In[7]:


digits.images[0].shape


# # Data Preprocessing 

# In[8]:


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# In[9]:


data[0]


# In[10]:


data[0].shape


# In[11]:


data.shape


# # Scaling Data

# In[12]:


data.min()


# In[13]:


data.max()


# In[14]:


data = data/16


# In[15]:


data.min()


# In[16]:


data.max()


# In[17]:


data[0]


# # Train test split Data

# In[18]:


xtrain, xtest, ytrain, ytest = train_test_split(data, digits.target, test_size = 0.3)


# In[19]:


xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# # Modeling

# In[20]:


rf = RandomForestClassifier()


# In[21]:


rf.fit(xtrain, ytrain)


# # Prediction

# In[22]:


y_pred = rf.predict(xtest)


# In[23]:


y_pred


# # Model Evaluation

# # Confusion Matrix

# In[24]:


confusion_matrix(ytest, y_pred)


# In[25]:


import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(ytest, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# # Classification Report

# In[26]:


print(classification_report(ytest, y_pred))


# In[27]:


from sklearn.metrics import classification_report
import pandas as pd

# Generate classification report
report = classification_report(ytest, y_pred, target_names=digits.target_names, output_dict=True)
report_digits = pd.DataFrame(report).transpose()

# Plot classification report
plt.figure(figsize=(10, 7))
sns.heatmap(report_digits.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Classification Report')
plt.show()


# # Correlation Matrix

# In[28]:


import seaborn as sns

# Assuming you have a DataFrame with features and target
# Compute the correlation matrix
correlation_matrix = pd.DataFrame(data).corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# # Accuracy Score

# In[29]:


# Predict labels for new data
# Predict for one observation
single_prediction = random_forest.predict(X_val[0].reshape(1, -1))
print(f"Prediction for one observation: {single_prediction}")

# Predict for multiple observations
multiple_predictions = random_forest.predict(X_val[:10])
print(f"Predictions for multiple observations: {multiple_predictions}")

# Make predictions on the entire validation set
predictions = random_forest.predict(X_val)
print(f"Predictions shape: {predictions.shape}")

# Calculate accuracy score
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_val, predictions)

# Plot confusion matrix with accuracy score in title
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Accuracy Score: {accuracy:.2f}', size=15)
plt.show()


# # Learning Curve

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Example using a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy', n_jobs=-1
)

# Mean and std deviation
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
plt.title('Learning Curves')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# # Explaination

# Handwritten Digit Prediction, or Digit Classification, involves training a computer to recognize and interpret handwritten numbers, much like how we identify digits ourselves. In this project, we start by collecting a set of images, each depicting a handwritten digit from 0 to 9. We divide these images into a training set, used to teach the computer, and a testing set, used to assess its performance. By applying a machine learning model, the computer learns to identify patterns and features in the digits during training. We then evaluate its accuracy by testing it on new, unseen images, comparing its predictions to the actual digits. This process enables the computer to accurately recognize handwritten digits, which can be applied in practical applications such as reading postal codes or interpreting handwritten documents, effectively teaching the computer to understand human handwriting.
