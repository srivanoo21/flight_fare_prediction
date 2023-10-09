import os
import sys
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.flight_fare_prediction.utils import read_sql_data
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train_data.csv')
    test_data_path:str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ## reading code from mysql
            #df = read_sql_data()
            df = pd.read_csv(os.path.join('dataset', 'raw_data.csv'))

            logging.info("Reading completed from mysql database")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
