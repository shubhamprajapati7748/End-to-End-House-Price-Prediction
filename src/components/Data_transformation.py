import os 
import sys 
import pandas as pd 
import numpy as np 

from dataclasses import dataclass 
from src.logger import logging
from src.exception import CustomException 

from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler 
from src.utils.utils import save_object 

from src.components.Data_ingestion import DataIngestion 
from src.components.Model_trainer import ModelTrainer
from src.components.Model_evaluation import ModelEvaluation

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_tranformation(sel, train_df : pd.DataFrame) -> ColumnTransformer:
        logging.info("Creating preprocessor object")

        numerical_columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

        num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())]
                )
        
        preprocessor = ColumnTransformer([
            ("numerical_pipeline", num_pipeline, numerical_columns)
        ])
        logging.info("Preprecessor object created")

        return preprocessor

        pass 

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test datasets completed")


            # Training dataset - Independent and dependent features
            target_column_name = "medv"
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            ## Test datasets - Independent and dependent features
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Train data before transformation \n {train_df}")
            logging.info(f"Test data before transformation \n {test_df}")

            ## Data transformation with preprocessor
            preprocessor_obj = self.get_data_tranformation(input_feature_train_df)
            input_feature_train_transform_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_transform_arr = preprocessor_obj.transform(input_feature_test_df)

            ## Final train and test arr with Independent and dependent features
            train_arr = np.c_[input_feature_train_transform_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_transform_arr, target_feature_test_df]

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("Saved the preprocessor object")
            logging.info("Data transformation completed")

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pass