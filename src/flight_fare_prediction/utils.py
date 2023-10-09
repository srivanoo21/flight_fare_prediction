import os
import sys
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV

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
        df = pd.read_sql_query('Select * from train_data', mydb)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e, sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def missing_col(data_train, data_test):
    '''
    This function adds missing columns from training dataset to testing dataset
    '''
    l1 = list(data_train.columns.values)
    l2 = list(data_test.columns.values)
    missing_col = [i for i in l1 if i not in l2]

    for i in missing_col:
        pos = data_train.columns.get_loc(i)
        data_test.insert(pos, i, None)
        data_test[i] = 0

    return data_train, data_test


def evaluate_models(xtrain, ytrain, xtest, ytest, models, param):
    try:
        report = {}
        #xtrain, xtest = np.array(xtrain), np.array(xtest)
        #ytrain, ytest = np.array(ytrain), np.array(ytest)

        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, scoring='neg_mean_squared_error', cv=3)
            logging.info("model training has been started")
            logging.info(f"train shape is {xtrain.shape, ytrain.shape}")    
            logging.info(f"test shape is {xtest.shape, ytest.shape}")    

            xtrain, ytrain = np.array(xtrain), np.array(ytrain)
            xtest, ytest = np.array(xtest), np.array(ytest)
            gs.fit(xtrain, np.ravel(ytrain))

            model.set_params(**gs.best_params_)
            model.fit(xtrain, ytrain)
            logging.info("model has been trained successfully")

            train_model_score = model.score(xtrain, ytrain)
            logging.info("training score has been generated")

            test_model_score = model.score(xtest, ytest)
            logging.info("testing score has been generated")

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
        