import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils.utils import save_object, evaluate_models
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR 
from catboost import CatBoostRegressor

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):

        try: 
            logging.info("Model training started")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(),
            }

            params = {
                    "DecisionTreeRegressor": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    },
                    "RandomForestRegressor":{
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Gradient Boosting":{
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "LinearRegression":{},
                    "Ridge" : {},
                    "Lasso" : {},
                    "KNeighborsRegressor" : {},
                    "CatBoostRegressor" : {},
                    "AdaBoostRegressor":{
                        'learning_rate':[.1,.01,0.5,.001], 
                        'n_estimators': [8,16,32,64,128,256]
                    } 
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model Report: {model_report}")

            ## to get best model score from report 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r_square = r2_score(y_test, predicted)
            logging.info(f'Best Model: {best_model}')
            logging.info(f"R2 score of best model is {r_square}")
            return r_square
    
        except Exception as e:
            raise CustomException(e, sys)