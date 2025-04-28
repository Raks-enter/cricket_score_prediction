import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter': ['best', 'random'],
                    #'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features': ['sqrt', 'log2', None],
                    #'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    #'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    #'criterion': ['squared_error', 'friedman_mse'],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    #'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "K-Nearest Neighbor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    #'algorithm': ['ball_tree', 'kd_tree', 'brute']
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    #'loss': ['linear', 'square', 'exponential'],
                    #'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            if not model_report:
                raise CustomException("No models could be evaluated successfully.")

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² score >= 0.6")

            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            return score

        except Exception as e:
            raise CustomException(e, sys)
