import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import ModelTrainer

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"Initiating DataIngestion")
        try:
            data = pd.read_csv('notebook\data\Churn_Modelling.csv')

            logging.info(f"Dataset read as dataframe")

            unrelevant_columns = ['RowNumber', 'CustomerId', 'Surname']
            data = data.drop(unrelevant_columns, axis=1)
            logging.info(f"Unrelevant columns: {unrelevant_columns} deleted")

            cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
            data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))
            logging.info(f"Datatype changed to categorical columns: {cat_cols}")
            logging.info(f"Datatype {data[cat_cols].dtypes}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path)

            train_set, test_set = train_test_split(data, test_size=0.2,
                                                   random_state=42, stratify= data['Exited'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Train and test data ingestion completed")
            logging.info(f"Training data shape: {train_set.columns}")
            logging.info(f"Test data shape: {test_set.columns}")
            return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr, test_arr)
