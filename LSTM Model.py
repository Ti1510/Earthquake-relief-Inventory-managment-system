#!/usr/bin/env python
# coding: utf-8

# ## To see Earthquake occured in past from past data

# In[ ]:


import pandas as pd
import folium

# Load the earthquake data
data = pd.read_csv("Indian_earthquake_data.csv")

# Create a Folium map centered around India
map_center = [20.5937, 78.9629]  # Approximate center of India
earthquake_map = folium.Map(location=map_center, zoom_start=4)

# Add earthquake locations to the map
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,  # Size of the marker
        color="red",  # Outline color
        fill=True,
        fill_color="red",
        fill_opacity=0.7,
        popup=f"Location: {row['Location']}<br>Magnitude: {row['Magnitude']}",
    ).add_to(earthquake_map)

earthquake_map


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[2]:


# Load earthquake dataset
df = pd.read_csv("Indian_earthquake_data.csv", parse_dates=['Origin Time'])

# Sort by date (important for time-series data)
df = df.sort_values(by="Origin Time")

# Select relevant features
features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
data = df[features]

# Normalize the data (LSTM works better with normalized input)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define prediction target (Magnitude)
target = scaled_data[:, -1]  # Last column (Magnitude)


# In[3]:


def create_sequences(data, target, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of past `n` records
        y.append(target[i + sequence_length])  # Next magnitude
    return np.array(X), np.array(y)

# Define sequence length (e.g., using last 10 earthquakes to predict the next)
sequence_length = 10
X, y = create_sequences(scaled_data, target, sequence_length)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape for LSTM (samples, time steps, features)
print(f"X_train shape: {X_train.shape}")  # Expected shape: (samples, sequence_length, features)


# In[4]:


# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)  # Predicting one value (Magnitude)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()


# In[5]:


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Time")
plt.show()


# In[6]:


# Predict on test data
y_pred = model.predict(X_test)

# Inverse transform predictions (convert back to original scale)
y_pred_original = scaler.inverse_transform(np.c_[np.zeros((y_pred.shape[0], 3)), y_pred])[:, -1]
y_test_original = scaler.inverse_transform(np.c_[np.zeros((y_test.shape[0], 3)), y_test])[:, -1]

# Plot actual vs predicted magnitudes
plt.figure(figsize=(10, 5))
plt.plot(y_test_original, label="Actual Magnitude")
plt.plot(y_pred_original, label="Predicted Magnitude", linestyle="dashed")
plt.legend()
plt.title("Earthquake Magnitude Prediction")
plt.show()


# In[7]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")


# In[8]:


# Get the last sequence from the dataset
last_sequence = scaled_data[-sequence_length:]

# Reshape for prediction
last_sequence = np.expand_dims(last_sequence, axis=0)  # Shape (1, sequence_length, features)

# Predict the next magnitude
future_magnitude = model.predict(last_sequence)

# Convert back to original scale
future_magnitude_original = scaler.inverse_transform(
    np.c_[np.zeros((1, 3)), future_magnitude]
)[:, -1][0]

print(f"Predicted Next Earthquake Magnitude: {future_magnitude_original}")


# In[ ]:




