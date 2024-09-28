import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_keras_model, load_object


class CustomData:
    def __init__(self, CreditScore:int, Geography:str, Gender:str, Age:int, Tenure:int, Balance:float,
                 NumOfProducts:int, HasCrCard:int, IsActiveMember:int,EstimatedSalary:float):
        self.CreditScore = CreditScore
        self.Geography=Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary


    def preprocess_features(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Tenure":[self.Tenure],
                "Balance": [self.Balance],
                "NumOfProducts": [self.NumOfProducts],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary]
            }

            features = pd.DataFrame(custom_data_input_dict)
            cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
            features[cat_cols] = features[cat_cols].apply(lambda x: x.astype('category'))
            return features

        except Exception as e:
            raise CustomException(e,sys)

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model = load_keras_model('artifacts/model.keras')
            preprocessor_path = 'artifacts/preprocessor.pkl'
            preprocessor = load_object(preprocessor_path)
            data = preprocessor.transform(features)
            probability = model.predict(data)
            class_prediction = (probability > 0.5).astype(int)
            prediction = class_prediction[0][0]
            return prediction
        except Exception as e:
            raise CustomException(e, sys)