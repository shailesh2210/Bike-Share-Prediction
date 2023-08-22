import os
import sys
sys.path.append(r'D:\live end to end machine learning projects')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts" , "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "raw.csv")

@dataclass
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has Started.....")

        try:
            df = pd.read_csv("data\Bikes.csv")
            logging.info("Reading dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(os.path.join(self.ingestion_config.raw_data_path) , header=True , index=False)

            train , test = train_test_split(df , test_size=0.2 , random_state=42)
            logging.info("train test split")

            train.to_csv(self.ingestion_config.train_data_path ,index=False , header = True)

            test.to_csv(self.ingestion_config.test_data_path , index = False , header = True)
            logging.info("Splitting data into train and test")

            return (
                self.ingestion_config.train_data_path ,
                self.ingestion_config.test_data_path
            )
            logging("Data Ingestion has Sucessfully completed!")
        except Exception as e:
            raise CustomException(sys, e)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()