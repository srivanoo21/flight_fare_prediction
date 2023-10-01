import os
import sys
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.flight_fare_prediction.utils import read_sql_data


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    #raw_data_pth:str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ## reading code from mysql
            #df1, df2 = read_sql_data()
            df1 = pd.read_csv(os.path.join('dataset', 'Data_Train.csv'))
            df2 = pd.read_csv(os.path.join('dataset', 'Data_Test.csv'))
            logging.info("Reading completed from mysql database")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df1.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df2.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
