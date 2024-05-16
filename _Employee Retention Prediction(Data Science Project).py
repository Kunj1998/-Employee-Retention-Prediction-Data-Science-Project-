#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES AND DATASETS

# In[75]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[76]:


employee_df=pd.read_csv("C:/Users/H.P/Downloads/Human_Resources.csv")
employee_df


# In[77]:


employee_df.head(5)


# In[78]:


employee_df.tail(10)


# In[79]:


employee_df.info()


# In[80]:


employee_df.describe()


# # VISUALIZE DATASET

# In[81]:


# Let's replace the 'Attritition' and 'overtime' column with integers before performing any visualizations 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for column in ['Over18', 'OverTime', 'Attrition']:
    employee_df[column] = encoder.fit_transform(employee_df[column])


# In[82]:


employee_df.head(4)


# In[83]:


# Let's see if we have any missing data, luckily we don't!
sns.heatmap(employee_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[84]:


employee_df.hist(bins = 30, figsize = (20,20), color = 'r')
# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavy
# It makes sense to drop 'EmployeeCount' and 'Standardhours' since they do not change from one employee to the other


# In[85]:


# It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other
# Let's drop 'EmployeeNumber' as well
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)


# In[86]:


employee_df


# In[87]:


# Let's see how many employees left the company! 
left_df        = employee_df[employee_df['Attrition'] == 1]
stayed_df      = employee_df[employee_df['Attrition'] == 0]


# In[88]:


# Count the number of employees who stayed and left
# It seems that we are dealing with an imbalanced dataset 

print("Total =", len(employee_df))

print("Number of employees who left the company =", len(left_df))
print("Percentage of employees who left the company =", 1.*len(left_df)/len(employee_df)*100.0, "%")
 
print("Number of employees who did not leave the company (stayed) =", len(stayed_df))
print("Percentage of employees who did not leave the company (stayed) =", 1.*len(stayed_df)/len(employee_df)*100.0, "%")


# In[89]:


left_df.describe()

# Let's compare the mean and std of the employees who stayed and left 
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home 
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option level


# In[90]:


stayed_df.describe()


# In[91]:


plt.figure(figsize=[25, 12])
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df)


# In[92]:


plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)
plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)
plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)
plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)

# Single employees tend to leave compared to married and divorced
# Sales Representitives tend to leave compared to any other job 
# Less involved employees tend to leave the company 
# Less experienced (low job level) tend to leave the company 


# In[93]:


# KDE (Kernel Density Estimate) is used for visualizing the Probability Density of a continuous variable. 
# KDE describes the probability density at different values in a continuous variable. 

plt.figure(figsize=(12,7))

sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home')


# In[94]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Years With Current Manager')


# In[95]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'b')

plt.xlabel('Total Working Years')


# In[96]:


# Let's see the Gender vs. Monthly Income
plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = employee_df)


# In[97]:


# Let's see the monthly income vs. job role
plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)


# # CREATE TESTING AND TRAINING DATASET & PERFORM DATA CLEANING
# 
# 
# 
# 

# In[98]:


employee_df.head(3)


# In[99]:


X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat


# In[100]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()


# In[101]:


X_cat.shape


# In[102]:


X_cat = pd.DataFrame(X_cat)
X_cat


# In[103]:


# note that we dropped the target 'Atrittion'
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]
X_numerical


# In[104]:


X_all = pd.concat([X_cat, X_numerical], axis = 1)
X_all


# In[105]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Convert column names to strings
X_all.columns = X_all.columns.astype(str)

X = scaler.fit_transform(X_all)


# In[106]:


X


# In[107]:


y = employee_df['Attrition']
y


# # TRAIN AND EVALUATE A LOGISTIC REGRESSION CLASSIFIER

# In[108]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[109]:


X_train.shape


# In[110]:


X_test.shape


# In[111]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[112]:


y_pred


# In[113]:


# Testing Set Performance
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)


# In[114]:


from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# In[115]:


print(classification_report(y_test, y_pred))


# # TRAIN AND EVALUATE A RANDOM FOREST CLASSIFIER

# In[116]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[117]:


y_pred = model.predict(X_test)


# In[118]:


from sklearn.metrics import confusion_matrix
# Testing Set Performance
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)


# In[119]:


print(classification_report(y_test, y_pred))


# In[120]:


print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# # TRAIN AND EVALUATE A DEEP LEARNING MODEL

# In[121]:


import tensorflow as tf


# In[122]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[123]:


model.summary()


# In[124]:


model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[125]:


# oversampler = SMOTE(random_state=0)
# smote_train, smote_target = oversampler.fit_sample(X_train, y_train)
# epochs_hist = model.fit(smote_train, smote_target, epochs = 100, batch_size = 50)
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50)


# In[126]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


# In[127]:


y_pred


# In[128]:


epochs_hist.history.keys()


# In[129]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])


# In[130]:


plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])


# In[131]:


# Testing Set Performance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


# In[132]:


print(classification_report(y_test, y_pred))


# In[133]:


print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# In[ ]:





# In[ ]:




