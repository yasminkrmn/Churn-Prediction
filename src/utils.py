import os
import sys
import dill
import tensorflow as tf
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)




def save_keras_model(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        model.save(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def load_keras_model(checkpoint_path):
    model = tf.keras.models.load_model(checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    status.expect_partial()
    return model

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
