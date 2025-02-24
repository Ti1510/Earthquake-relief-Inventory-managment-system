#!/usr/bin/env python
# coding: utf-8

# ## Predict the magnitude of the next earthquake with fixed location

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[26]:


# Load earthquake dataset
df = pd.read_csv("Indian_earthquake_data.csv", parse_dates=['Origin Time'])

# Sort data by date
df = df.sort_values(by="Origin Time")

# Select relevant features
features = ['Latitude', 'Longitude', 'Depth']
target = ['Magnitude']

# Extract feature and target values
X = df[features].values
y = df[target].values

# Standardize the features (KNN is distance-based, so scaling is important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")


# In[28]:


# Define the KNN model with k=5 (default choice, can be tuned)
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict on test data
y_pred = knn_model.predict(X_test)


# In[29]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")


# In[30]:


plt.figure(figsize=(25, 25))
plt.plot(y_test, label="Actual Magnitude", marker="o")
plt.plot(y_pred, label="Predicted Magnitude", marker="x", linestyle="dashed")
plt.legend()
plt.title("KNN Earthquake Magnitude Prediction")
plt.show()


# In[31]:


errors = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    errors.append(mean_absolute_error(y_test, y_pred_k))

# Plot error vs. k values
plt.figure(figsize=(8, 5))
plt.plot(k_values, errors, marker="o")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Absolute Error")
plt.title("Choosing Optimal k in KNN")
plt.show()


# In[32]:


# Predicting an earthquake at a new location
new_earthquake = np.array([[35.0, 138.0, 20.0]])  # Example (Latitude, Longitude, Depth)
new_earthquake_scaled = scaler.transform(new_earthquake)

predicted_magnitude = knn_model.predict(new_earthquake_scaled)
print(f"Predicted Magnitude for New Earthquake: {predicted_magnitude[0][0]}")


# ## risk prone zone or not

# In[13]:


# Load earthquake dataset
df = pd.read_csv("Indian_earthquake_data.csv", parse_dates=['Origin Time'])

# Sort data by date
df = df.sort_values(by="Origin Time")

# Define risk labels: High-Risk (1) if magnitude >= 5.0, else Low-Risk (0)
df['Risk_Label'] = (df['Magnitude'] >= 5.0).astype(int)

# Select features and target
features = ['Latitude', 'Longitude', 'Depth']
target = 'Risk_Label'

X = df[features].values
y = df[target].values

# Normalize the features for better KNN performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
# Define the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict on test data
y_pred = knn_model.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[19]:


errors = []
k_values = range(1, 21)  # Try k from 1 to 20

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    errors.append(1 - accuracy_score(y_test, y_pred_k))  # Error rate

# Plot error vs. k values
plt.figure(figsize=(8, 5))
plt.plot(k_values, errors, marker="o", linestyle="dashed", color="r")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Error Rate")
plt.title("Choosing Optimal k in KNN Classification")
plt.show()


# In[20]:


#Predict earthquake risk for a new location
new_location = np.array([[35.0, 138.0, 15.0]])  
new_location_scaled = scaler.transform(new_location)

predicted_risk = knn_model.predict(new_location_scaled)
risk_label = "High-Risk Zone" if predicted_risk[0] == 1 else "Low-Risk Zone"

print(f"Predicted Risk for New Location: {risk_label}")


# In[ ]:




