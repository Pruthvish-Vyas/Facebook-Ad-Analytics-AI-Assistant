import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        rideable_type: str,
        started_at: str,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        member_casual: str
    ):
        self.rideable_type = rideable_type
        self.started_at = started_at
        self.start_lat = start_lat
        self.start_lng = start_lng
        self.end_lat = end_lat
        self.end_lng = end_lng
        self.member_casual = member_casual

    def get_data_as_data_frame(self):
        try:
            # Parse the datetime
            started_at_dt = pd.to_datetime(self.started_at)
            
            # Extract features
            day_of_week = started_at_dt.day_name()
            hour_of_day = started_at_dt.hour
            
            # Create a dummy ended_at for the dataframe (not used in prediction)
            # This is just to maintain the structure expected by the preprocessor
            ended_at = started_at_dt
            
            custom_data_input_dict = {
                "ride_id": ["dummy_id"],  # Placeholder
                "rideable_type": [self.rideable_type],
                "started_at": [started_at_dt],
                "ended_at": [ended_at],
                "start_station_name": [None],  # Can be None as we'll impute
                "start_station_id": [None],
                "end_station_name": [None],
                "end_station_id": [None],
                "start_lat": [self.start_lat],
                "start_lng": [self.start_lng],
                "end_lat": [self.end_lat],
                "end_lng": [self.end_lng],
                "member_casual": [self.member_casual],
                "day_of_week": [day_of_week],
                "hour_of_day": [hour_of_day],
                # ride_duration_minutes will be calculated during preprocessing
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
