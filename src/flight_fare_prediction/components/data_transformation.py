import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.logger import logging
import os
import re
import datetime as dt


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

       
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info("Train and test file has been loaded")

            logging.info("Preprocessing the training dataset")

            train_data.dropna(inplace=True)

            train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y').dt.day
            train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y').dt.month
            train_data.drop(labels=['Date_of_Journey'], axis=1, inplace=True)
            
            train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
            train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
            train_data.drop(labels=['Dep_Time'], axis=1, inplace=True)

            train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
            train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
            train_data.drop(labels=['Arrival_Time'], axis=1, inplace=True)

            for duration in train_data['Duration']:
                if 'h' not in duration:
                    j = train_data.loc[train_data['Duration']==duration].index[0]
                    train_data['Duration'].iloc[j] = "0h " + duration
                    numerical_columns = ["writing_score", "reading_score"]

            duration_hours = []
            duration_min = []

            for duration in train_data['Duration']:
                duration = re.sub('[^0-9]', ' ', duration)
                dur = duration.split()
                if len(dur) > 1:
                    duration_hours.append(int(dur[0]))
                    duration_min.append(int(dur[1]))
                else:
                    duration_hours.append(int(dur[0]))
                    duration_min.append(0)

            # Adding columns 'Duration_hrs' and 'Duration_min' in dataset train_data
            train_data['Duration_hrs'] = duration_hours
            train_data['Duration_min'] = duration_min
            train_data.drop(labels=['Duration'], axis=1, inplace=True)

            logging.info("Date time columns has been encoded for training data")

            Airline = train_data[['Airline']]
            Airline = pd.get_dummies(Airline, drop_first=True)
            Source = train_data[['Source']]
            Source = pd.get_dummies(Source, drop_first=True)
            Destination = train_data[['Destination']]
            Destination = pd.get_dummies(Destination, drop_first=True)
            
            train_data.drop(labels=['Additional_Info', 'Route'], axis=1, inplace=True)

            encode = {
                'non-stop': 0,
                '1 stop': 1,
                '2 stops': 2,
                '3 stops': 3,
                '4 stops': 4,
            }
            train_data['Total_Stops'] = train_data['Total_Stops'].apply(lambda x: encode[x])

            data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)
            data_train.drop(labels=['Airline', 'Source', 'Destination'], axis=1, inplace=True)

            logging.info("Categorical columns has now been encoded")
            logging.info(f"Shape of training data: , {data_train.shape}")


            logging.info("Preprocessing the test dataset")
        
            test_data.dropna(inplace=True)
        
            # Date Of Journey
            test_data['Journey_day'] = pd.to_datetime(test_data['Date_of_Journey'], format='%d/%m/%Y').dt.day
            test_data['Journey_month'] = pd.to_datetime(test_data['Date_of_Journey'], format='%d/%m/%Y').dt.month
            test_data.drop(labels=['Date_of_Journey'], axis=1, inplace=True)

            # Dep Time
            test_data['Dep_hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour
            test_data['Dep_min'] = pd.to_datetime(test_data['Dep_Time']).dt.minute
            test_data.drop(labels=['Dep_Time'], axis=1, inplace=True)

            # Arrival Time
            test_data['Arrival_hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour
            test_data['Arrival_min'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute
            test_data.drop(labels=['Arrival_Time'], axis=1, inplace=True)

            # Duration
            for duration in test_data['Duration']:
                if 'h' not in duration:
                    j = test_data.loc[test_data['Duration']==duration].index[0]
                    test_data['Duration'].iloc[j] = "0h " + duration

            duration_hours = []
            duration_min = []

            for duration in test_data['Duration']:
                duration = re.sub('[^0-9]', ' ', duration)
                dur = duration.split()
                # Add hours and minutes
                if len(dur) > 1:
                    duration_hours.append(int(dur[0]))
                    duration_min.append(int(dur[1]))
                # Add minutes as 0 where we have only hours
                else:
                    duration_hours.append(int(dur[0]))
                    duration_min.append(0)
                
            test_data['Duration_hrs'] = duration_hours
            test_data['Duration_min'] = duration_min
            test_data.drop(labels=['Duration'], axis=1, inplace=True)

            logging.info("Date time columns has been encoded for testing data")

            # Categorical Data
            Airline = test_data[['Airline']]
            Airline = pd.get_dummies(Airline, drop_first=True)
            Source = test_data[['Source']]
            Source = pd.get_dummies(Source, drop_first=True)
            Destination = test_data[['Destination']]
            Destination = pd.get_dummies(Destination, drop_first=True)

            test_data.drop(labels=['Additional_Info', 'Route'], axis=1, inplace=True)

            # Total Stops
            encode = {
                'non-stop': 0,
                '1 stop': 1,
                '2 stops': 2,
                '3 stops': 3,
                '4 stops': 4,
            }
            test_data['Total_Stops'] = test_data['Total_Stops'].apply(lambda x: encode[x])

            # Concatenate dataframe --> test_data + Airline + Source + Destination
            data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)
            data_test.drop(labels=['Airline', 'Source', 'Destination'], axis=1, inplace=True)

            logging.info("Categorical columns has now been encoded")
            logging.info(f"Shape of testing data: , {data_test.shape}")


            logging.info("Now returning the transformed training and testing dataset")

            ## To do: add a preprocessor object in return parameters and save it in .pkl file
            return (
                data_train,
                data_test
            )
        
        except Exception as e:
            raise CustomException(sys,e)