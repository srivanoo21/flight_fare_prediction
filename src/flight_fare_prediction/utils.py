import os
import sys
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started...")
    try:
        mydb = pymysql.connect(
            host=host, user=user, password=password, db=db
        )
        logging.info("Connection established...", mydb)
        df1 = pd.read_sql_query('Select * from train_data', mydb)
        df2 = pd.read_sql_query('Select * from test_data', mydb)
        print(df1.head())
        print(df2.head())

        return df1, df2

    except Exception as e:
        raise CustomException(e, sys)


