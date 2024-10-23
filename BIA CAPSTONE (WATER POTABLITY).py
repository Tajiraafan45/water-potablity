#!/usr/bin/env python
# coding: utf-8

# # STEP 1 -IMPORT ALL NECESSARY LIBRARIES

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')  


# # STEP 2 -IMPORT DATA AND PRE PROCESSING 

# In[22]:


data = pd.read_csv("water_potability.csv")


# In[23]:


data.head()


# In[24]:


data.shape #To check the shape of the data


# In[25]:


data.isnull().sum() #To check if there are any missing values


# In[26]:


data.info()


# In[27]:


# Impute missing values with median
data['ph'].fillna(data['ph'].median(), inplace=True)
data['Sulfate'].fillna(data['Sulfate'].median(), inplace=True)
data['Trihalomethanes'].fillna(data['Trihalomethanes'].median(), inplace=True)


# In[28]:


# Verify no missing values
data.isnull().sum()


# # step 3 - EDA

# In[29]:


# Visualize distributions
plt.figure(figsize=(15, 12))
for i, column in enumerate(data.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[31]:


# Visualize outliers using boxplots
plt.figure(figsize=(15, 12))
for i, column in enumerate(data.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


# # STEP 4 - SPLIT THE DATA

# In[32]:


# Feature Scaling
scaler = StandardScaler()
X = data.drop('Potability', axis=1)
y = data['Potability']


# In[33]:


X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# In[34]:


# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)


# In[35]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# In[36]:


X_train.head()


# In[40]:


# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)}


# In[41]:


# Training and cross-validation
model_scores = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model_scores[name] = np.mean(cv_scores)
    print(f"{name} CV Accuracy: {np.mean(cv_scores):.4f}")


# In[46]:


# Hyperparameter tuning for Random Forest as an example
param_grid = {
    'n_estimators': [ 200],
    'max_depth': [ 30],
    'min_samples_split': [2]
}


# In[47]:


grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# In[48]:


print("\nBest parameters for Random Forest:")
print(grid_search.best_params_)


# In[49]:


# Evaluate the tuned Random Forest model
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)


# In[50]:


# Evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")


# In[51]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[52]:


# ROC Curve and AUC Score
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


# In[53]:


plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[54]:


# Feature importance for tree-based models
importances = best_rf.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()


# In[55]:


print("\nProject completed successfully!")


# In[ ]:




