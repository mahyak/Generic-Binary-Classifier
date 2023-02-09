from typing import Protocol
import pandas
import numpy
from dataclasses import dataclass


class SamplingStrategy(Protocol):
    def create_sample_set(self, population: pandas.DataFrame, sample_size: int) -> pandas.DataFrame:
        """Return sample set from a pupulationset."""

class SystematicSampling(SamplingStrategy):
    def create_sample_set(self, population: pandas.DataFrame, sample_size: int) -> pandas.DataFrame:
        population_size = len(population)
        chosen_idx = numpy.arange(1, population_size, step = population_size/(sample_size+92))
        sample_set = population.iloc[chosen_idx]
        return sample_set
    
@dataclass
class NonsubscribersSampling():
    population: pandas.DataFrame
    sample_size: int

    def apply_sampling(self, samplin_strategy: SamplingStrategy) -> pandas.DataFrame:
        return samplin_strategy.create_sample_set(self.population, self.sample_size)