import os
import sys
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
import mlflow.keras

from kerastuner.tuners import RandomSearch
from mlflow.models.signature import infer_signature
from src.exception import CustomException
from src.logger import logging
from src.utils import save_keras_model
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    trained_model_obj_file_path = os.path.join('artifacts', 'model.keras')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                        activation='relu', input_shape=(self.input_shape,)))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            self.input_shape = X_train.shape[1]
            logging.info(f'X_train.shape:  {X_train.shape}')
            logging.info(f'X_test.shape:  {X_test.shape}')
            logging.info(f'y_train.shape:  {y_train.shape}')
            logging.info(f'y_test.shape:  {y_test.shape}')

            mlflow.tensorflow.autolog()

            tuner = RandomSearch(
                self.build_model,
                objective='val_accuracy',
                max_trials=10,
                executions_per_trial=1,
                directory='hyperparameter_tuning',
                project_name='churn_model'
            )

            tuner.search_space_summary()

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            with mlflow.start_run():
                logging.info(f"Starting hyperparameter tuning...")
                tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=64,
                             callbacks=[early_stopping])

                logging.info(f"Hyperparameter tuning completed.")
                best_model = tuner.get_best_models(num_models=1)[0]
                logging.info('Best model found for train and test')
                logging.info(f'Best model: {best_model.summary()}')

                predictions = best_model.predict(X_train)

                input_data = X_train.astype('float32')
                signature = infer_signature(input_data, predictions)

                loss, accuracy = best_model.evaluate(X_test, y_test)
                mlflow.log_metric('test_loss', loss)
                mlflow.log_metric('test_accuracy', accuracy)
                logging.info(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

                mlflow.keras.log_model(
                    best_model,
                    "model",
                    signature=signature
                )

                save_keras_model(
                    file_path=self.model_trainer_config.trained_model_obj_file_path,
                    model=best_model
                )

        except Exception as e:

            raise CustomException(e, sys)

