import os
import sys
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except CustomException as e:
        raise CustomException(e, sys)


import os
import sys
import dill
import mlflow
import mlflow.keras

from src.exception import CustomException


def save_object(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except CustomException as e:
        raise CustomException(e, sys)
