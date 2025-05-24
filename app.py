import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging

# Title of the Streamlit app
st.title("Bike Ride Duration Prediction")
st.write("This app predicts the duration of a bike ride based on various features.")

# Input fields for user data
st.sidebar.header("Input Features")

# Bike type selection
rideable_type = st.sidebar.selectbox(
    "Bike Type", 
    ["electric_bike", "classic_bike", "docked_bike"]
)

# Date and time selection
start_date = st.sidebar.date_input("Start Date", datetime.now())
start_time = st.sidebar.time_input("Start Time", datetime.now().time())

# Combine date and time
started_at = datetime.combine(start_date, start_time)

# Location inputs
st.sidebar.subheader("Start Location")
start_lat = st.sidebar.number_input("Start Latitude", 41.65, 45.64, 41.89)
start_lng = st.sidebar.number_input("Start Longitude", -87.83, -73.80, -87.65)

st.sidebar.subheader("End Location")
end_lat = st.sidebar.number_input("End Latitude", 41.65, 42.07, 41.90)
end_lng = st.sidebar.number_input("End Longitude", -87.83, -87.52, -87.65)

# Membership status
member_casual = st.sidebar.selectbox("Membership Status", ["member", "casual"])

# Display a map with the start and end points
st.subheader("Ride Map")
map_data = pd.DataFrame({
    'lat': [start_lat, end_lat],
    'lon': [start_lng, end_lng],
    'label': ['Start', 'End']
})
st.map(map_data)

# Button to make predictions
if st.button("Predict Ride Duration"):
    try:
        # Format the datetime for the model
        started_at_str = started_at.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a CustomData instance with user inputs
        custom_data = CustomData(
            rideable_type=rideable_type,
            started_at=started_at_str,
            start_lat=start_lat,
            start_lng=start_lng,
            end_lat=end_lat,
            end_lng=end_lng,
            member_casual=member_casual
        )

        # Convert the input data to a DataFrame
        input_data = custom_data.get_data_as_data_frame()

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()

        # Make predictions
        with st.spinner('Calculating prediction...'):
            predictions = predict_pipeline.predict(input_data)

        # Display the prediction
        minutes = predictions[0]
        hours, remainder = divmod(minutes, 60)
        
        st.success(f"The predicted ride duration is: {int(hours)} hours and {int(remainder)} minutes")
        
        # Show additional information
        st.info(f"Total duration in minutes: {minutes:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    logging.info('Prediction completed')
