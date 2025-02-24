#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import mysql.connector
import pandas as pd
import folium
from math import sqrt
from datetime import datetime
from streamlit_folium import folium_static

# Location database with coordinates
LOCATION_COORDINATES = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kochi": (9.9312, 76.2673),
    "Bhopal": (23.2599, 77.4126),
    "Nagpur": (21.1458, 79.0882),
    "Visakhapatnam": (17.6868, 83.2185),
    "Patna": (25.5941, 85.1376)
}

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="dams_user",
        password="dams123",
        database="DAMS"
    )

def get_total_resources():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(food_available), SUM(water_available), SUM(medicine_available) FROM available")
    total = cursor.fetchone()
    conn.close()
    
    # Convert None values to 0
    if total[0] is None:
        return (0, 0, 0)
    return total

def add_available_resources(location, food, water, medicine):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = """INSERT INTO available (location, food_available, water_available, medicine_available)
             VALUES (%s, %s, %s, %s)"""
    values = (location, food, water, medicine)
    cursor.execute(sql, values)
    conn.commit()
    conn.close()

def check_available_resources(food_required, water_required, medicine_required):
    """
    Check if there are sufficient resources available to fulfill the requirement.
    Returns tuple (bool, dict) indicating if resources are available and current available amounts
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                COALESCE(SUM(food_available), 0) as total_food,
                COALESCE(SUM(water_available), 0) as total_water,
                COALESCE(SUM(medicine_available), 0) as total_medicine
            FROM available
            WHERE food_available > 0 
            OR water_available > 0 
            OR medicine_available > 0
        """)
        available = cursor.fetchone()
        
        if available is None:
            return False, {'food': 0, 'water': 0, 'medicine': 0}
            
        resources_available = {
            'food': float(available[0]),
            'water': float(available[1]),
            'medicine': float(available[2])
        }
        
        is_available = (resources_available['food'] >= food_required and 
                      resources_available['water'] >= water_required and 
                      resources_available['medicine'] >= medicine_required)
        
        return is_available, resources_available
    finally:
        conn.close()

def fulfill_requirement(food_required, water_required, medicine_required):
    """
    Fulfill the requirement by subtracting from available resources
    Returns True if successful, False otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get available resources ordered by timestamp (FIFO)
        cursor.execute("""
            SELECT id, food_available, water_available, medicine_available
            FROM available
            WHERE food_available > 0 
            OR water_available > 0 
            OR medicine_available > 0
            ORDER BY timestamp
        """)
        available_resources = cursor.fetchall()
        
        remaining_food = food_required
        remaining_water = water_required
        remaining_medicine = medicine_required
        
        # Keep track of updates to make
        updates = []
        
        for resource in available_resources:
            resource_id = resource[0]
            curr_food = resource[1]
            curr_water = resource[2]
            curr_medicine = resource[3]
            
            # Calculate how much to take from current resource
            food_taken = min(remaining_food, curr_food)
            water_taken = min(remaining_water, curr_water)
            medicine_taken = min(remaining_medicine, curr_medicine)
            
            # Add to updates list
            updates.append((
                curr_food - food_taken,
                curr_water - water_taken,
                curr_medicine - medicine_taken,
                resource_id
            ))
            
            # Update remaining requirements
            remaining_food -= food_taken
            remaining_water -= water_taken
            remaining_medicine -= medicine_taken
            
            # If all requirements are fulfilled, break
            if remaining_food <= 0 and remaining_water <= 0 and remaining_medicine <= 0:
                break
        
        # Apply all updates in a single transaction
        for update in updates:
            cursor.execute("""
                UPDATE available
                SET food_available = %s,
                    water_available = %s,
                    medicine_available = %s
                WHERE id = %s
            """, update)
        
        # Clean up any empty resources
        cursor.execute("""
            DELETE FROM available
            WHERE food_available <= 0 
            AND water_available <= 0 
            AND medicine_available <= 0
        """)
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Error fulfilling requirement: {e}")
        return False
    finally:
        conn.close()

def request_aid(location, latitude, longitude, food, water, medicine):
    """
    Adds the aid request to requirements table and attempts to fulfill it.
    Returns a tuple of (success, message, resources_available)
    """
    # First check if resources are available
    is_available, current_resources = check_available_resources(food, water, medicine)
    
    if not is_available:
        return False, "Insufficient resources available.", current_resources
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Add requirement to database
        sql = """INSERT INTO requirement (location, food_required, water_required, medicine_required, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s)"""
        values = (location, food, water, medicine, latitude, longitude)
        cursor.execute(sql, values)
        conn.commit()
        
        # Attempt to fulfill the requirement
        if fulfill_requirement(food, water, medicine):
            return True, "Aid request added and fulfilled successfully!", current_resources
        else:
            return False, "Aid request added but could not be fulfilled automatically.", current_resources
    except Exception as e:
        conn.rollback()
        return False, f"Error processing aid request: {e}", current_resources
    finally:
        conn.close()

def calculate_eoq(demand, ordering_cost, holding_cost):
    return sqrt((2 * demand * ordering_cost) / holding_cost) if demand > 0 and ordering_cost > 0 and holding_cost > 0 else 0

def visualize_disaster_location(latitude, longitude, location):
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    folium.Marker([latitude, longitude], popup=f"Disaster Location: {location}", icon=folium.Icon(color='red')).add_to(m)
    return m

def show_resource_history():
    """Display a history of all aid requests and their status"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            r.id,
            r.food_required, 
            r.water_required, 
            r.medicine_required,
            r.timestamp
        FROM requirement r
        ORDER BY r.timestamp DESC
    """)
    requirements = cursor.fetchall()
    conn.close()
    
    if requirements:
        requirement_df = pd.DataFrame(requirements, 
                                     columns=['ID', 'Food Required (tons)', 
                                              'Water Required (kL)', 'Medicine Required (units)', 
                                              'Timestamp'])
        st.dataframe(requirement_df)
    else:
        st.info("No aid requests found in the database.")

# Main Streamlit App
st.title("🌍 Disaster Relief Inventory Management System")
menu = ["Request Aid", "Provide Aid", "View Resources", "Request History"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "View Resources":
    st.subheader("📊 Total Available Resources")
    total = get_total_resources()
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Food", value=f"{total[0]:.2f} tons")
    
    with col2:
        st.metric(label="Water", value=f"{total[1]:.2f} kL")
    
    with col3:
        st.metric(label="Medicine", value=f"{total[2]:.2f} units")
    
    # Add a more detailed breakdown by storage location
    st.subheader("Resource Breakdown by Location")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            location,
            food_available,
            water_available,
            medicine_available,
            timestamp
        FROM available
        WHERE food_available > 0 
        OR water_available > 0 
        OR medicine_available > 0
        ORDER BY timestamp DESC
    """)
    locations = cursor.fetchall()
    conn.close()
    
    if locations:
        location_df = pd.DataFrame(locations, 
                                  columns=['Location', 'Food (tons)', 'Water (kL)', 
                                           'Medicine (units)', 'Added On'])
        st.dataframe(location_df)
    else:
        st.warning("No resources available in any location!")

elif choice == "Provide Aid":
    st.subheader("🚚 Provide Aid (Add Available Resources)")
    
    with st.form("resource_form"):
        # For the provide aid form, use a selectbox with option to type custom location
        location_options = ["Custom"] + sorted(list(LOCATION_COORDINATES.keys()))
        location_choice = st.selectbox("Select Location", location_options)
        
        # Only show the text input field if "Custom" is selected
        if location_choice == "Custom":
            location = st.text_input("Enter Location Name")
        else:
            location = location_choice
            
        food = st.number_input("Food Available (tons)", min_value=0.0, step=0.1)
        water = st.number_input("Water Available (kiloliters)", min_value=0.0, step=0.1)
        medicine = st.number_input("Medicine Available (units)", min_value=0, step=1)
        submitted = st.form_submit_button("Add Resources")
        
        if submitted:
            # Validate that a location is provided if "Custom" is selected
            if location_choice == "Custom" and not location.strip():
                st.error("Please enter a location name.")
            else:
                add_available_resources(location, food, water, medicine)
                st.success("✅ Resources added successfully!")
                
                # Show updated totals
                total = get_total_resources()
                st.write("### Updated Resource Totals:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Food", value=f"{total[0]:.2f} tons")
                with col2:
                    st.metric(label="Water", value=f"{total[1]:.2f} kL")
                with col3:
                    st.metric(label="Medicine", value=f"{total[2]:.2f} units")

elif choice == "Request Aid":
    st.subheader("🚨 Request Aid (Add New Requirement)")
    
    # Create a Streamlit session state to store coordinates
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 0.0
    if 'longitude' not in st.session_state:
        st.session_state.longitude = 0.0
    
    # Allow location selection and coordinate update before the form
    location_options = ["Custom"] + sorted(list(LOCATION_COORDINATES.keys()))
    location_choice = st.selectbox("Select Disaster Location", location_options, key="location_choice")
    
    # Update coordinates based on location selection
    if location_choice != "Custom" and location_choice in LOCATION_COORDINATES:
        st.session_state.latitude, st.session_state.longitude = LOCATION_COORDINATES[location_choice]
    
    with st.form("request_form"):
        if location_choice == "Custom":
            location = st.text_input("Enter Location Name")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
            with col2:
                longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
        else:
            location = location_choice
            # Use session state values for coordinates
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", -90.0, 90.0, value=st.session_state.latitude)
            with col2:
                longitude = st.number_input("Longitude", -180.0, 180.0, value=st.session_state.longitude)
            
        st.subheader("Required Resources")
        col1, col2, col3 = st.columns(3)
        with col1:
            food = st.number_input("Food (tons)", min_value=0.0, step=0.1)
        with col2:
            water = st.number_input("Water (kL)", min_value=0.0, step=0.1)
        with col3:
            medicine = st.number_input("Medicine (units)", min_value=0, step=1)
        
        st.subheader("EOQ Parameters")
        col1, col2 = st.columns(2)
        with col1:
            ordering_cost = st.number_input("Ordering Cost", min_value=0.1, value=100.0, step=10.0)
        with col2:
            holding_cost = st.number_input("Holding Cost", min_value=0.1, value=10.0, step=1.0)
        
        submitted = st.form_submit_button("Request Aid")
        
        if submitted:
            success, message, resources = request_aid(location, latitude, longitude, food, water, medicine)
            
            if success:
                st.success(message)
            else:
                st.error(message)
                st.write("### Current Available Resources:")
                st.write(f"🍞 Food: {resources['food']:.2f} tons")
                st.write(f"💧 Water: {resources['water']:.2f} kiloliters")
                st.write(f"💊 Medicine: {resources['medicine']:.2f} units")
            
            # If request was added (whether fulfilled or not), show EOQ and map
            if "added" in message:
                # Calculate EOQ
                food_eoq = calculate_eoq(food, ordering_cost, holding_cost)
                water_eoq = calculate_eoq(water, ordering_cost, holding_cost)
                medicine_eoq = calculate_eoq(medicine, ordering_cost, holding_cost)
                
                st.write("### 📈 Economic Order Quantity (EOQ) Calculations:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Food EOQ", f"{food_eoq:.2f} tons")
                with col2:
                    st.metric("Water EOQ", f"{water_eoq:.2f} kL")
                with col3:
                    st.metric("Medicine EOQ", f"{medicine_eoq:.2f} units")
                
                # Show map
                st.subheader("📍 Disaster Location")
                map_display = visualize_disaster_location(latitude, longitude, location)
                folium_static(map_display)

elif choice == "Request History":
    st.subheader("📜 Aid Request History")
    show_resource_history()

