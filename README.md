# Earthquake-relief-Inventory-managment-system
# Research Methodology for Earthquake Prediction and Disaster Management System

# 1. Research Overview

This study brings together machine learning for predicting earthquakes and a practical system for managing disasters. The approach includes three main parts:
- Using K-Means clustering and LSTM neural networks to predict earthquakes
- Visualizing historical earthquake data
- Setting up a system to manage disaster relief supplies

# 2. Data Collection and Preprocessing

# 2.1 Dataset
- **Primary data source:** Indian earthquake data from the file "Indian_earthquake_data.csv"
- **Key features collected:**
  - Origin time
  - Latitude and longitude
  - Depth of the earthquake
  - Magnitude
  - Location information

# 2.2 Data Preprocessing Steps
- Sort the earthquake data by origin time.
- Use StandardScaler to standardize the features for the K-Means model.
- Apply MinMax scaling to normalize the input data for the LSTM model.
- Create sequences of data for time series analysis.

# 3. Predictive Modeling Approach

# 3.1 K-Means Based Prediction
**Objective:** Predict earthquake magnitude and spot high-risk areas  
**How it works:**
- **Features used:** Latitude, Longitude, and Depth
- **Data split:** 80% for training, 20% for testing
- **Model:** Use KNeighborsRegressor to predict magnitude
- **Risk Classification:** Areas are labeled as High-Risk or Low-Risk based on a magnitude threshold of 5.0
- **Tuning:** Choose the best k-value by looking at the error rates

# 3.2 LSTM Neural Network Implementation
**Objective:** Predict future earthquake magnitudes over time  
**Network Structure:**
- **Input Layer:** Takes data from the last 10 earthquakes
- **LSTM Layers:** Two layers with 50 units each
- **Dropout Layers:** 0.2 dropout rate to prevent overfitting
- **Dense Layer:** For the final prediction

**Training Settings:**
- **Loss Function:** Mean Squared Error
- **Optimizer:** Adam
- **Batch Size:** 32
- **Epochs:** 50

# 4. Disaster Management System Design

# 4.1 System Architecture
- Use a web-based interface with Streamlit.
- Integrate a MySQL database for managing resources.
- Add interactive maps using Folium.

# 4.2 Core Functionalities
**Resource Management:**
- Track available resources like food, water, and medicine in real-time.
- Distribute resources based on location.
- Calculate Economic Order Quantity (EOQ).

**Aid Request Processing:**
- Map the geographic location of requests.
- Assess the resources needed.
- Automatically allocate resources.

**Monitoring and Visualization:**
- Show disaster locations on interactive maps.
- Display dashboards for resource availability.
- Track past aid requests

# 5. Evaluation Metrics

# 5.1 Earthquake Prediction Models
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Classification metrics for risk assessment:
  - Accuracy
  - Confusion Matrix
  - Classification Report

# 5.2 Disaster Management System
- How efficiently resources are allocated
- Analysis of geographic coverage
- Response time metrics
- Rates of resource utilization

# 6. Implementation Tools and Technologies

**Programming Language:** Python 3.x  
**Machine Learning Libraries:**  
- scikit-learn for K-Means  
- TensorFlow for LSTM neural networks  
- Pandas for data manipulation  

**Visualization Tools:**  
- Matplotlib for performance analysis  
- Folium for geographic visualization  

**Web Framework:** Streamlit  
**Database:** MySQL
