import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation for Facebook advertising data
        '''
        try:
            # Define columns based on the Facebook advertising dataset
            numerical_columns = [
                "campaign_id", "fb_campaign_id", "interest1", "interest2", "interest3",
                "impressions", "clicks", "spent", "total_conversion"
            ]
            categorical_columns = ["age", "gender"]

            # Pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ],
                remainder='drop'  # Drop other columns
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train dataset shape: {train_df.shape}")
            logging.info(f"Test dataset shape: {test_df.shape}")
            
            # Print column names for debugging
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            
            # Feature engineering for Facebook advertising data
            for df in [train_df, test_df]:
                # Convert date columns to datetime
                df['reporting_start'] = pd.to_datetime(df['reporting_start'])
                df['reporting_end'] = pd.to_datetime(df['reporting_end'])
                
                # Calculate campaign duration in days
                df['campaign_duration_days'] = (df['reporting_end'] - df['reporting_start']).dt.days + 1
                
                # Calculate CTR (Click Through Rate)
                df['ctr'] = df['clicks'] / df['impressions']
                df['ctr'] = df['ctr'].fillna(0)
                
                # Calculate CPC (Cost Per Click)
                df['cpc'] = df['spent'] / df['clicks']
                df['cpc'] = df['cpc'].fillna(0)
                df['cpc'] = df['cpc'].replace([np.inf, -np.inf], 0)
                
                # Calculate conversion rate
                df['conversion_rate'] = df['approved_conversion'] / df['clicks']
                df['conversion_rate'] = df['conversion_rate'].fillna(0)
                df['conversion_rate'] = df['conversion_rate'].replace([np.inf, -np.inf], 0)

            logging.info("Feature engineering completed")
            
            # Define target column - we'll predict approved_conversion
            target_column_name = "approved_conversion"
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Input feature train shape after transform: {input_feature_train_arr.shape}")
            
            # Convert target arrays to numpy arrays and reshape if needed
            target_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_test_arr = np.array(target_feature_test_df).reshape(-1, 1)
            
            # Concatenate features and target
            train_arr = np.hstack((input_feature_train_arr, target_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_test_arr))
            
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            logging.info(f"Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
