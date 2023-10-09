import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.logger import logging
from src.flight_fare_prediction.utils import save_object, evaluate_models, missing_col


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Splitting the data into xtrain, ytrain, xtest, ytest")
            xtrain, ytrain, xtest, ytest = (
                train_data.drop(labels=['Price'], axis=1),
                train_data[['Price']],
                test_data.drop(labels=['Price'], axis=1),
                test_data[['Price']],
            )

            logging.info("data is splitted into xtrain, ytrain, xtest, ytest")    
            xtrain, xtest = missing_col(xtrain, xtest)
            logging.info("missing columns have been adjusted")

            models = {
               # "Random Forest": RandomForestRegressor(),
              #  "Decision Tree": DecisionTreeRegressor(),
              #  "Gradient Boosting": GradientBoostingRegressor(),
              #  "Linear Regression": LinearRegression(),
              #  "XG Boost": XGBRegressor(),
              #  "AdaBoost Regressor": AdaBoostRegressor(),
                "KNN Regressor": KNeighborsRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_features': ['sqrt', 'log2']
                },
                "Random Forest": {
                    'n_estimators': [50, 60, 70, 100, 120, 140],
                    'max_depth': [4, 6, 8, 10, 12],
                    'min_samples_leaf': [1, 2, 3, 4, 5]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [50, 60, 70, 100, 120, 140]
                },
                "Linear Regression": {},
                "XG Boost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [50, 60, 70, 100, 120, 140]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [50, 60, 70, 100, 120, 140]
                },
                "KNN Regressor": {
                    'n_neighbors': [4, 5, 6, 7, 8, 9, 10]
                }
                    
            }

            model_report:dict = evaluate_models(xtrain, ytrain, xtest, ytest, models, params)
            logging.info("models are now evaluated")

            # To get best model score from the dict
            best_model_score = max(model_report.values())
            logging.info(f"best model score is fetched from dictionary i.e. {best_model_score}")

            # To get best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                                                  ]     
            logging.info(f"best model name is fetched from dictionary i.e. {best_model_name}")

            best_model = models[best_model_name]
            logging.info("best model is now stored")

            print("This is the best model:")
            print(best_model_name)

            model_names = list(params.keys())
            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/srivanoo21/flight_fare_prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow

            with mlflow.start_run():
                
                preds = best_model.predict(xtest)
                (rmse, mae, r2) = self.eval_metrics(ytest, preds)
                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Mode registry does not work with file store
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")


            # To do - correct error for this one
            #if best_model_score < 0.6:
                #raise CustomException("No best model found", sys)
            logging.info(f"Best model has been found - {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model {best_model_name} has been saved")
            
            predicted = best_model.predict(xtest)
            r2_square = r2_score(ytest, predicted)

            logging.info(f"The r2 score is {r2_square}")
            
            return r2_square


        except Exception as e:
            raise CustomException(e, sys)