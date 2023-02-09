from abc import ABC, abstractmethod
from typing import List
import pandas
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

@dataclass
class CategoricalFeatures:
    dataframe: pandas.DataFrame

    def extract_categorical_columns(self) -> List:
        """
        Return list of categorical columns in a dataframe.
        """
        categorical_columns = list(set(self.dataframe.columns) - set(self.dataframe._get_numeric_data().columns))
        return categorical_columns

class CategoricalStrategy(ABC):
    @abstractmethod
    def create_to_numerical(self, dataframe: pandas.DataFrame, column_name: str) -> pandas.DataFrame:
        dataframe = pandas.get_dummies(dataframe, columns=[f'{column_name}'], prefix=f'{column_name}_is')
        return dataframe

class LabelEncoderStrategy(CategoricalStrategy):
    def create_to_numerical(self, dataframe: pandas.DataFrame, column_name: str) -> pandas.DataFrame:
        label_encoder = LabelEncoder()
        dataframe[f'{column_name}'] = label_encoder.fit_transform(dataframe[f'{column_name}'])
        return dataframe

class GetDummiestrategy(CategoricalStrategy):
    def create_to_numerical(self, dataframe: pandas.DataFrame, column_name: str) -> pandas.DataFrame:
        label_encoder = LabelEncoder()
        dataframe[f'{column_name}'] = label_encoder.fit_transform(dataframe[f'{column_name}'])
        return dataframe       

@dataclass
class CategoricalToNumerical:
    training_set: pandas.DataFrame
    
    def convert_categorical_to_numberical(self, convert_strategy: CategoricalStrategy, column_name: str) -> pandas.DataFrame:
        return convert_strategy.create_to_numerical(self.training_set, column_name)



    