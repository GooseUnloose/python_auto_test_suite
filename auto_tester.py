import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV

import os,multiprocessing,math


def sum_parameter_configurations(parameters:dict):
    
    parameter_sum = 1
    
    for parameter in parameters.values():
        parameter_sum = parameter_sum * len(parameter)
    
    return parameter_sum
    


#Returns a list of core utilisation for each layer of parallelism
def parallel_layer_distribution(df:pd.DataFrame, 
                                parameters:dict, 
                                cv:int, 
                                cpu_cores:int = 0, 
                                layer_count: int = 3, 
                                core_overhead_factor:float = 3.0) -> list[int]:
    '''Determines parellisation layer distribution
    
    keyword arguments:
    
    df -- dataframe for testing
    cpu_cores -- number of CPU Cores the system has (default = 0)
    '''
    
    if type(df) != pd.DataFrame:
        raise ValueError("df Must be a pd.DataFrame object")
    
    #One core is reserved for the device's operating system, whilst the others are listed as avaliable cores for parallelsation
    total_cpu_count = cpu_cores
    avaliable_cpu_count = total_cpu_count - 1
    
    #Parameter tuning layers
    parameter_permutations = sum_parameter_configurations(parameters)
    
    parameter_workers = min(avaliable_cpu_count,parameter_permutations)
    cores_per_parameter = math.floor(avaliable_cpu_count / parameter_workers)
    
    
    return cores_per_parameter




params = {
    'test1':[1,2],
    'test2':[1],
}

sum_parameter_configurations(params)
parallel_layer_distribution(pd.DataFrame(),params,cv=5,cpu_cores=4)
    