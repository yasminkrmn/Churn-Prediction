import os
import sys

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_obj(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            cat_columns = [col for col in data.columns if data[col].dtype == 'category']
            num_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            num_columns = [col for col in num_columns if col != 'Exited']

            cat_pipeline = Pipeline(
                    steps=[
                        ('onehotencoder', OneHotEncoder(drop='first', sparse_output=False))
                    ]
                )

            num_pipeline = Pipeline(
                    steps=[
                        ('robustscaler', RobustScaler())
                    ]
                )

            preprocessor = ColumnTransformer(
                    transformers=[

                        ("ohe_cat_pipeline", cat_pipeline, cat_columns),
                        ("num_pipeline", num_pipeline, num_columns)
                    ]
                )

            logging.info(f'Pipeline processed')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info(f'Imported train and test dataset')

            preprocessing_obj = self.get_transformation_obj(train_data)
            target_col = 'Exited'

            X_train = train_data.drop(target_col, axis=1)
            y_train = train_data[target_col]

            X_test = test_data.drop(target_col, axis=1)
            y_test = test_data[target_col]

            logging.info(
                f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}'
                f' X_test shape: {X_test.shape}, y_test shape: {y_test.shape}'
            )
            X_train_arr = preprocessing_obj.fit_transform(X_train) #Relook
            X_test_arr = preprocessing_obj.transform(X_test) #Relook

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            logging.info(
                f"Training set shape:{train_arr.shape}"
                f"Test set shape: {test_arr.shape}"
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)



