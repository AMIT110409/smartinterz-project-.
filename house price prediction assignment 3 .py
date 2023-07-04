#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Step 0: Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[4]:


#Step 1: Load the dataset
df = pd.read_csv('Housing.csv')


# In[26]:


#Perform Univariate Analysis
plt.figure(figsize=(12, 6))

# Histogram of Price
plt.subplot(2, 2, 1)
plt.hist(df['price'], bins=20, edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Histogram of Area
plt.subplot(2, 2, 2)
plt.hist(df['area'], bins=20, edgecolor='black')
plt.xlabel('Area')
plt.ylabel('Frequency')

# Countplot of Bedrooms
plt.subplot(2, 2, 3)
sns.countplot(data=df, x='bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')

# Countplot of Bathrooms
plt.subplot(2, 2, 4)
sns.countplot(data=df, x='bathrooms')
plt.xlabel('Bathrooms')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[10]:


#Step 3: Performing Analysis
#Perform Bivariate Analysis
plt.figure(figsize=(12, 6))

# Scatter plot of Area vs. Price
plt.subplot(1, 2, 1)
plt.scatter(df['area'], df['price'])
plt.xlabel('Area')
plt.ylabel('Price')

# Box plot of Furnishing Status vs. Price
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='furnishingstatus', y='price')
plt.xlabel('Furnishing Status')
plt.ylabel('Price')

plt.tight_layout()
plt.show()


# In[11]:


#Perform Multivariate Analysis
plt.figure(figsize=(10, 8))

# Heatmap of correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.show()


# In[12]:


# Step 4: Perform descriptive statistics
descriptive_stats = df.describe()
print(descriptive_stats)


# In[15]:


# Step 5: Check for Missing values and deal with them
missing_values = df.isnull().sum()
print(missing_values)


# In[16]:


# Step 6: Find and replace outliers - Code not provided


# In[17]:


# Step 7: Check for Categorical columns and perform encoding
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


# In[19]:


# Step 8: Split the data into dependent and independent variables
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']


# In[20]:


# Step 9: Scale the independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[21]:


# Step 10: Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[22]:


# Step 11: Build the Model
model = LinearRegression()


# In[23]:


# Step 12: Train the Model
model.fit(X_train, y_train)


# In[24]:


# Step 13: Test the Model
y_pred = model.predict(X_test)


# In[25]:


# Step 14: Measure the performance using Metrics
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:





# In[ ]:




