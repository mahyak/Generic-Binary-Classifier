from typing import Protocol, Any
import numpy
import pandas
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from dataclasses import dataclass


grid = { 
    "n_estimators":[1, 2, 4, 8, 16, 32, 64, 100, 200],
    "max_depth": numpy.linspace(1, 32, 32, endpoint=True),
    "learning_rate":[1, 0.5, 0.25, 0.1, 0.05, 0.01],
    "min_samples_split": numpy.linspace(0.1, 1.0, 10, endpoint=True),
    "min_samples_leaf": numpy.linspace(0.1, 0.5, 5, endpoint=True),
    "max_features": list(range(1,14))}

class HyperparametersTuningTechnique(Protocol):
    def apply_tunning_technique(X_train: pandas.DataFrame, y_train: pandas.DataFrame, base_model: Any) -> dict:
        """Returns best parameters."""

class RandomizedSearchTechnique(HyperparametersTuningTechnique):
    def apply_tunning_technique(X_train: pandas.DataFrame, y_train: pandas.DataFrame, base_model: Any) -> dict:
        randomized_search = RandomizedSearchCV(estimator=base_model,
                                               param_distributions=grid,
                                               n_iter=15, 
                                               cv=5,
                                               verbose=2,
                                               random_state=42, 
                                               n_jobs=4)
        randomized_search.fit(X_train, y_train)
        return randomized_search.best_params_
    

class RandomizedSearchTechnique(HyperparametersTuningTechnique):
    def apply_tunning_technique(X_train: pandas.DataFrame, y_train: pandas.DataFrame, base_model: Any) -> dict:
        grid_search =  GridSearchCV(estimator = base_model, 
                                    param_grid = grid,
                                    cv = 5, 
                                    n_jobs = 8, 
                                    verbose = 2)
    
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

@dataclass
class HyperparameteTuning:
    X_train: pandas.DataFrame
    y_train: pandas.DataFrame
    selected_model: Any

    def add_tunned_hyperparameters(self, tuning_technique: HyperparametersTuningTechnique):
        return tuning_technique.apply_tunning_technique(self.X_train, self.y_train, self.selected_model)

