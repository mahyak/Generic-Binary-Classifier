from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier as XGboost, RandomForestClassifier
from dataclasses import dataclass
import pandas
from sklearn.model_selection import  cross_val_score, KFold
from typing import List, Any
import pathlib

import pickle

classification_models = {
    'Logistic Regression': LogisticRegression(), 
    'Linear Svm': SVC(kernel='linear'), 
    'Radial Svm': SVC(kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=11), 
    'DT': DecisionTreeClassifier(), 
    'Random forest': RandomForestClassifier(), 
    'XGBoost': XGboost()
    }

models = {
    'Logistic Regression': LogisticRegression, 
    'Linear Svm': SVC, 
    'Radial Svm': SVC,
    'KNN': KNeighborsClassifier, 
    'DT': DecisionTreeClassifier, 
    'Random forest': RandomForestClassifier, 
    'XGBoost': XGboost
    }


DATA_DIR = "./data"
SAVE_MODEL_PATH = pathlib.Path(f"{DATA_DIR}/final_model.sav")

@dataclass
class ModelSelection:
    X_train: pandas.DataFrame
    y_train: pandas.DataFrame

    def apply_model(self) -> List[float]:
        kfold = KFold(n_splits=10, random_state=22, shuffle=True)
        accuracy = [(cross_val_score(cls_model, self.X_train, self.y_train, cv = kfold, scoring = "accuracy")).mean() for cls_model in classification_models.values()]
        return accuracy
    
    @staticmethod
    def select_best_model(accuracy: List[float]) -> pandas.DataFrame:
        models_accuracy_dataframe = pandas.DataFrame(accuracy, index=classification_models.keys())   
        models_accuracy_dataframe.columns=['CV Mean']   
        sorted_model_by_accuracy = models_accuracy_dataframe.sort_values(['CV Mean'], ascending=[0])
        best_model = models[sorted_model_by_accuracy.index[0]]
        return sorted_model_by_accuracy, best_model




class TrainSelectedModel:
    def apply_model(self, selected_model: Any, tunned_hyperparams: dict, X_train: pandas.DataFrame, y_train: pandas.DataFrame) -> Any:
        model = selected_model(
            learning_rate=tunned_hyperparams['learning_rate'], 
            max_features=tunned_hyperparams['max_features'], 
            max_depth = tunned_hyperparams['max_depth'], 
            min_samples_leaf = tunned_hyperparams['min_samples_leaf'], 
            min_samples_split = tunned_hyperparams['min_samples_split'], 
            n_estimators = tunned_hyperparams['n_estimators']
            )
        model.fit(X_train, y_train)
        return model
    
    @staticmethod
    def save_model(model: Any):
        triangelSelectModel = SAVE_MODEL_PATH
        pickle.dump(model, open(triangelSelectModel, 'wb'))

  

        
        

        