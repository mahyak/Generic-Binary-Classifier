import pathlib
from ds.feature_importance import apply_feature_importance
from ds.categorical import CategoricalToNumerical, CategoricalFeatures, LabelEncoderStrategy, GetDummiestrategy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ds.model_selectionl import ModelSelection, TrainSelectedModel
from ds.hyperparameter_tuning import HyperparameteTuning, RandomizedSearchTechnique
import pandas
from typing import Any
from sklearn.metrics import classification_report, accuracy_score


# Data configuration
DATA_DIR = "./data"
TRIANGLESELECT_DATA = pathlib.Path(f"{DATA_DIR}/prospective_customers.csv")


class BinaryClassClassifier:
    @staticmethod
    def separate_dependent_column(df: pandas.DataFrame):
        # Separate dependent colum 
        X = df.loc[:, df.columns != 'tssubscriber']
        X = X.loc[:, X.columns != 'city']
        y = df['tssubscriber']
        return X, y
    
    @staticmethod
    def split_train_dev_sets(X: pandas.DataFrame, y: pandas.DataFrame):
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.8)
        return X_train, X_dev, y_train, y_dev 
    
    @staticmethod
    def convert_categorical_to_numerica(X: pandas.DataFrame) -> pandas.DataFrame:
            categorical_columns = CategoricalFeatures(X).extract_categorical_columns()
            to_numerical = CategoricalToNumerical(X)
            for categorical_col in categorical_columns:
                if categorical_col != "age_seg":
                    X = to_numerical.convert_categorical_to_numberical(LabelEncoderStrategy(), categorical_col)
                    if categorical_col == 'segment_type':
                        segment_type = list(X['segment_type'].unique())
                        X['segment_type_cat'] = X['segment_type'].apply(lambda x: segment_type.index(x))
                else:
                    X = to_numerical.convert_categorical_to_numberical(GetDummiestrategy(), categorical_col)
            return X
    
    @staticmethod
    def select_features(X: pandas.DataFrame) -> pandas.DataFrame:
        selected_features = apply_feature_importance(X)
        X_ = X[selected_features]
        return X_
    
    @staticmethod
    def scale_data(X):
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        return scaled_X
    
    @staticmethod
    def select_model(X_train: pandas.DataFrame, y_train: pandas.DataFrame):
        model = ModelSelection(X_train, y_train)
        models_accuracy = model.apply_model()
        model_result , best_model = model.select_best_model(models_accuracy)
        selected_model = best_model
        print(f"Accuracy of the models: {model_result}")
        return selected_model
    
    @staticmethod
    def tune_hyperparameters(X_train: pandas.DataFrame, y_train: pandas.DataFrame, selected_model: Any) -> dict:
        hyper_tunning = HyperparameteTuning(X_train, y_train, selected_model)
        tunned_hyperparams = hyper_tunning.add_tunned_hyperparameters(RandomizedSearchTechnique)
        return tunned_hyperparams
    
    @staticmethod
    def finalize_model(selected_model: Any, hparams: dict, X_train: pandas.DataFrame, y_train: pandas.DataFrame) -> Any:
        # Train Selected Model
        train_selected_model = TrainSelectedModel()
        finalize_model = train_selected_model.apply_model(selected_model=selected_model, tunned_hyperparams=hparams, X_train=X_train, y_train=y_train)
        train_selected_model.save_model(finalize_model)
        return finalize_model
    
    @staticmethod
    def apply_model(model: Any, X_dev: pandas.DataFrame, y_dev: pandas.DataFrame):
        predicted_y_dev = model.predict(X_dev)
        print(f"classification Repost: {classification_report(y_dev, predicted_y_dev)}")
        print(f"Accuracy Score is : {accuracy_score(y_dev, predicted_y_dev)}")
    
