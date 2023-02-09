import pandas
from pandas import DataFrame
from pathlib import Path



def load_data(file_path: Path) -> DataFrame:
    df = pandas.read_csv(file_path)
    return df
    
