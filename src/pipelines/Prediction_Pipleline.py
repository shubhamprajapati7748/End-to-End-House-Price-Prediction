import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self, features):
        try:
            model_path = os.path.join("Artifacts", "model.pkl")
            preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e, sys) 


class CustomData:
    def __init__(self, 
            crim : float, 
            zn : float, 
            indus : float, 
            chas : float, 
            nox : float, 
            rm : float, 
            age : float, 
            dis : float, 
            rad : float, 
            tax : float, 
            ptratio : float, 
            b : float, 
            lstat : float,     
        ):

        self.crim = crim
        self.zn = zn 
        self.indus = indus
        self.chas = chas 
        self.nox = nox
        self.rm = rm 
        self.age = age 

        self.rad = rad
        self.dis = dis 
        self.tax = tax
        self.ptratio = ptratio 
        self.lstat = lstat
        self.b = b 

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "crim" : [self.crim],
                "zn" : [self.zn],
                "indus" : [self.indus],
                "chas" : [self.chas],
                "nox" : [self.nox],
                "rm" : [self.rm],
                "age" : [self.age],
                "rad" : [self.age],
                "dis" : [self.dis],
                "tax" : [self.tax],
                "ptratio" : [self.ptratio],
                "lstat" : [self.lstat],
                "b" : self.b
            }
            return pd.DataFrame(custom_data_input_dict
                                )
        except Exception as e:
            raise CustomException(e,sys) 

