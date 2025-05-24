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
        campaign_id: int,
        fb_campaign_id: int,
        age: str,
        gender: str,
        interest1: int,
        interest2: int,
        interest3: int,
        impressions: float,
        clicks: int,
        spent: float,
        total_conversion: float,
        reporting_start: str,
        reporting_end: str
    ):
        self.campaign_id = campaign_id
        self.fb_campaign_id = fb_campaign_id
        self.age = age
        self.gender = gender
        self.interest1 = interest1
        self.interest2 = interest2
        self.interest3 = interest3
        self.impressions = impressions
        self.clicks = clicks
        self.spent = spent
        self.total_conversion = total_conversion
        self.reporting_start = reporting_start
        self.reporting_end = reporting_end

    def get_data_as_data_frame(self):
        try:
            # Parse the datetime
            reporting_start_dt = pd.to_datetime(self.reporting_start)
            reporting_end_dt = pd.to_datetime(self.reporting_end)
            
            # Calculate derived features
            campaign_duration_days = (reporting_end_dt - reporting_start_dt).days + 1
            ctr = self.clicks / self.impressions if self.impressions > 0 else 0
            cpc = self.spent / self.clicks if self.clicks > 0 else 0
            conversion_rate = 0  # We don't know approved_conversion for prediction
            
            custom_data_input_dict = {
                "ad_id": [1],  # Placeholder
                "reporting_start": [reporting_start_dt],
                "reporting_end": [reporting_end_dt],
                "campaign_id": [self.campaign_id],
                "fb_campaign_id": [self.fb_campaign_id],
                "age": [self.age],
                "gender": [self.gender],
                "interest1": [self.interest1],
                "interest2": [self.interest2],
                "interest3": [self.interest3],
                "impressions": [self.impressions],
                "clicks": [self.clicks],
                "spent": [self.spent],
                "total_conversion": [self.total_conversion],
                "campaign_duration_days": [campaign_duration_days],
                "ctr": [ctr],
                "cpc": [cpc],
                "conversion_rate": [conversion_rate]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
