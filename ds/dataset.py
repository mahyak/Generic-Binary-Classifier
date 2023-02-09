from ds.load_data import load_data
from ds.sampling import NonsubscribersSampling, SystematicSampling
import pandas
import pathlib

DATA_DIR = "./data"
POSITIVE_DATA = pathlib.Path(f"{DATA_DIR}/positive_examples.csv")
NEGATIVE_DATA = pathlib.Path(f"{DATA_DIR}/negative_examples.csv")

class TrainingSet:
    def processs_subscriber(self) -> pandas.DataFrame:
        positives = load_data(POSITIVE_DATA)
        return positives
    
    def processs_nonsubscriber(self) -> pandas.DataFrame:
        negative = load_data(NEGATIVE_DATA)
        return negative
    
def create_trainingset() -> pandas.DataFrame:
    training_set = TrainingSet()
    positives = training_set.processs_subscriber()
    negative = training_set.processs_nonsubscriber()
    negative_sampling = NonsubscribersSampling(negative, len(positives))
    negative_syatematic_sampling = negative_sampling.apply_sampling(SystematicSampling())
    training_df = pandas.concat([positives, negative_syatematic_sampling])
    training_df = training_df.set_index('id')
    return training_df
